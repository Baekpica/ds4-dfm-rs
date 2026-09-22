//! Common serving contract: requested / effective / qualified.
//!
//! Names are shared. Implementation cost and verification stay per family.
//! Forced options that a family cannot run become errors, not silent fallback.

use crate::identify::Identified;
use crate::shape::{ModelFamily, Shape, Variant, SHAPE_INKLING_SMALL, SHAPE_QWEN38_FLASH_NEXT};
use crate::Backend;
use serde_json::{json, Value};
use std::ffi::OsStr;
use std::fmt::{self, Write as _};

pub const DEFAULT_MEM_FLOOR_GB: u64 = 4;
pub const DEFAULT_MAX_SEQS: u32 = 2;
pub const DEFAULT_CTX: i32 = 8192;
pub const DEFAULT_SCHED_CHUNK: u32 = 4096;
/// C `bg_prefill_chunk_tokens` caps the boot chunk here unless
/// `DS4_CONT_PREFILL_NOFENCE=1`. The plan resolves the same cap so it cannot
/// advertise a yield the scheduler will not use.
pub const PREFILL_CHUNK_FENCE: u32 = 8192;
/// Scheduler yields the fence allows. Not native workspace/graph capacity.
/// 0 is the C one-shot / interleave-off sentinel, not a member of this set.
pub const VERIFIED_PREFILL_CHUNKS: [u32; 6] = [256, 512, 1024, 2048, 4096, 8192];
const GIB: u64 = 1 << 30;
const DOTS3_MAX_DRAFT: i32 = 3;
/// C `QWEN4EXP_YARN_MAX_FACTOR`: how far Qwen's RoPE context may stretch
/// before `ds4_session_create` refuses the context outright.
pub const QWEN_YARN_MAX_FACTOR: u32 = 4;
pub const DEFAULT_SCHED_LIVE: u32 = 512;
/// C `DS4_SERVER_PERSIST_MIN_TOKENS`: how much a continuous bank must hold
/// before retirement persists it. Not the disk store's record minimum.
pub const DEFAULT_BANK_PERSIST: i32 = 8192;

/// User-facing reuse policy. `Auto` is the best qualified path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixReuse {
    Off,
    Exact,
    Partial,
    Auto,
}

/// Sidecar/activation policy. Weights, enablement, and draft length stay separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpMode {
    Off,
    Auto,
    On,
}

/// What the native open and session creation require of the host. `Graph`
/// families refuse CPU but run on Metal; `Cuda` families refuse both, and
/// also refuse a distributed slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostNeed {
    Any,
    Graph,
    Cuda,
}

/// Whether the continuous lane may serve requests at all. `Serial` is the
/// legacy `DS4_SERVER_CONTINUOUS=0` switch: banks may still exist for the
/// static lane, but no request enters the bank driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneMode {
    Auto,
    Serial,
}

/// Whether this process is one full model or a distributed slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distribution {
    Single,
    Sliced,
}

/// Whether the native prefill fence applies. `DS4_CONT_PREFILL_NOFENCE=1`
/// lifts it, so the plan must read the same switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkFence {
    On,
    Off,
}

/// Concurrent banks/sequences, not context length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxSeqs {
    Auto,
    /// Legacy `--cont-width 0` / `DS4_SERVER_COALESCE_MAX=0`: serial, no banks.
    Off,
    Fixed(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IssueLevel {
    Error,
    Warn,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanIssue {
    pub level: IssueLevel,
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    None,
    Present,
    Qualified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankLane {
    Serial,
    OptIn,
    Persistent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpKind {
    None,
    Embedded,
    Sidecar,
    BoundOnly,
    DeepSeek,
}

/// Where speculative decode executes for a family. `Serial` families also
/// speculate on `NativeDecode`; `Bank` families only speculate inside the
/// continuous driver, so a disabled lane removes the feature entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecLane {
    None,
    Serial,
    Bank,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReuseKind {
    None,
    Exact,
    Partial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServingCaps {
    pub family: ModelFamily,
    pub variant: Variant,
    pub banks: BankLane,
    pub bank_support: Support,
    pub reuse: ReuseKind,
    pub reuse_support: Support,
    pub disk: Support,
    pub snapshot: Support,
    pub mtp: MtpKind,
    pub mtp_support: Support,
    pub spec_lane: SpecLane,
    /// Draft width below which the engine allocates no speculative runtime.
    pub spec_draft_min: i32,
    /// The host `model_open` and `ds4_session_create` insist on.
    pub host: HostNeed,
    /// Hard runtime maximum from `ds4_session_create`, not a qualification
    /// bound: above it the session cannot be created at all.
    pub ctx_max: Option<u32>,
    pub qualified_ctx: Option<u32>,
    pub qualified_banks: Option<u32>,
    pub qualified_prompt: Option<u32>,
    pub media_serial: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingRequest {
    pub prefix_reuse: PrefixReuse,
    pub mtp_mode: MtpMode,
    pub max_seqs: MaxSeqs,
    pub ctx: i32,
    pub mem_floor_gb: u64,
    pub kv_disk_dir: Option<String>,
    pub kv_disk_space_mb: Option<u64>,
    /// Disk store record minimum (`--kv-cache-min-tokens`).
    pub kv_min_tokens: Option<i32>,
    /// Continuous-bank persistence threshold (`DS4_SERVER_PERSIST_MIN_TOKENS`).
    pub bank_persist_min: Option<i32>,
    pub mtp_path: Option<String>,
    pub mtp_draft: Option<i32>,
    pub sched_chunk: Option<u32>,
    pub sched_chunk_live: Option<u32>,
    pub native_chunk: Option<u32>,
    pub print_plan: bool,
    pub check_config: bool,
    pub backend: Backend,
    pub chunk_fence: ChunkFence,
    pub distribution: Distribution,
    pub lane: LaneMode,
}

/// Facts known only after identify or engine open.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EngineFacts {
    pub mtp_loaded: bool,
    pub vision_loaded: bool,
    pub banks_fitted: Option<u32>,
    pub seq_cap: Option<u32>,
    pub disk_ready: Option<bool>,
    pub mtp_path_ok: Option<bool>,
    /// `Some(false)` when the named `--vision` artifact cannot attach.
    pub vision_path_ok: Option<bool>,
    /// `Some(false)` when the base artifact cannot load at all.
    pub artifact_ok: Option<bool>,
    /// `Some(false)` when a DSpark drafter cannot attach to this family.
    pub dspark_ok: Option<bool>,
    /// An IPC-dependent quote awaits successful native import during open.
    pub ipc_pending: bool,
    /// Actual native drafter import result after open; overrides manifest intent.
    pub drafter_shared: Option<bool>,
    /// `Some(false)` once the native fit refused the continuous lane.
    pub cont_lane: Option<bool>,
    /// `Some(false)` when the opened runtime has no partial checkpoint store.
    pub partial_reuse: Option<bool>,
    /// Native workspace/graph max. Distinct from scheduler yield.
    pub native_chunk: Option<u32>,
    pub shared_weights_bytes: Option<u64>,
    pub per_bank_bytes: Option<u64>,
    pub mtp_state_bytes: Option<u64>,
    pub scratch_bytes: Option<u64>,
    pub checkpoint_pool_bytes: Option<u64>,
    pub ple_bytes: Option<u64>,
    pub media_reserve_bytes: Option<u64>,
    /// Lazy media retained by each bank beyond the first; never resident credit.
    pub media_per_extra_bank_bytes: Option<u64>,
    /// Native fit reserve, including its floor and transient burst allowance.
    pub fit_headroom_bytes: Option<u64>,
    /// When set, `max-seqs=auto` must fit the named budgets under this ceiling.
    pub host_available_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestedView {
    pub lane: LaneMode,
    pub prefix_reuse: PrefixReuse,
    pub mtp_mode: MtpMode,
    pub max_seqs: MaxSeqs,
    pub ctx: i32,
    pub mem_floor_gb: u64,
    pub disk_dir: bool,
    pub mtp_path: bool,
    pub backend: Backend,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectiveView {
    pub prefix_reuse: ReuseKind,
    pub mtp_mode: MtpMode,
    pub mtp_weights: bool,
    pub mtp_draft: Option<i32>,
    pub max_seqs: u32,
    pub ctx: i32,
    pub mem_floor_gb: u64,
    pub disk: bool,
    pub banks_opt_in: bool,
    pub sched_chunk: u32,
    pub sched_chunk_live: u32,
    pub bank_persist_min: i32,
    pub disk_min_tokens: Option<i32>,
    pub native_chunk: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualifiedView {
    pub prefix_reuse: Support,
    pub disk: Support,
    pub mtp: Support,
    pub banks: Support,
    pub ctx: Option<u32>,
    pub banks_n: Option<u32>,
    pub prompt: Option<u32>,
    pub note: &'static str,
}

/// Memory the auto width has to host. Unit tests feed synthetic engine facts
/// so the arithmetic does not open a GGUF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServingQuote {
    pub shared_weights: u64,
    pub per_bank: u64,
    pub mtp_state: u64,
    pub scratch: u64,
    pub checkpoint_pool: u64,
    pub ple: u64,
    pub media_reserve: u64,
    pub floor: u64,
    pub available: u64,
    pub banks: u32,
    pub total: u64,
}

impl ServingQuote {
    fn cost(self, banks: u32) -> u64 {
        self.shared_weights
            .saturating_add(self.per_bank.saturating_mul(u64::from(banks)))
            .saturating_add(self.mtp_state)
            .saturating_add(self.scratch)
            .saturating_add(self.checkpoint_pool)
            .saturating_add(self.ple)
            .saturating_add(self.media_reserve)
            .saturating_add(self.floor)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPlan {
    pub family: Option<ModelFamily>,
    pub variant: Option<Variant>,
    pub family_name: &'static str,
    pub caps: Option<ServingCaps>,
    pub requested: RequestedView,
    pub effective: EffectiveView,
    pub qualified: QualifiedView,
    pub quote: Option<ServingQuote>,
    pub issues: Vec<PlanIssue>,
}

/// Host reuse decision after token LCP. Family restore is separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReusePath {
    Cold,
    Exact,
    Partial { ckpt: u32, replay: u32 },
}

pub fn host_reuse(lcp: u32, source_end: u32, ckpt: Option<u32>) -> ReusePath {
    if lcp == 0 {
        return ReusePath::Cold;
    }
    if lcp == source_end {
        return ReusePath::Exact;
    }
    match ckpt {
        Some(pos) if pos > 0 && pos <= lcp => ReusePath::Partial {
            ckpt: pos,
            replay: lcp - pos,
        },
        _ => ReusePath::Cold,
    }
}

/// What the host actually did to reuse KV, recorded where the decision is
/// made. Counters cannot tell these apart: an exact-frontier append also
/// prefills the new turn, and a fork looks like any other prefix hit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReuseTaken {
    /// Nothing reused.
    #[default]
    Cold,
    /// Reused a state ending at this prompt's common prefix; only the
    /// appended suffix is prefilled.
    Exact,
    /// Restored a checkpoint below the common prefix and replayed the gap.
    Partial,
    /// Copied another bank's state, preserving the source.
    Fork,
}

/// Why a stored or live candidate was refused. Recorded at the decision, so
/// an operator reads "the template dropped a block" instead of assuming the
/// disk store is broken. The strings are the contract's miss reasons.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReuseMiss {
    #[default]
    None,
    NoCheckpoint,
    RenderedPrefix,
    BelowThreshold,
    PayloadMismatch,
    StateReplay,
}

impl ReuseMiss {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::NoCheckpoint => "no checkpoint at or below LCP",
            Self::RenderedPrefix => "rendered prefix changed",
            Self::BelowThreshold => "below minimum token threshold",
            Self::PayloadMismatch => "payload family/layout mismatch",
            Self::StateReplay => "session state requires prefix replay",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestTrace {
    pub effective_lane: &'static str,
    pub reuse_kind: ReuseTaken,
    pub reuse_miss: ReuseMiss,
    pub speculation_active: bool,
    pub fallback_reason: Option<String>,
}

impl Default for ServingRequest {
    fn default() -> Self {
        Self {
            prefix_reuse: PrefixReuse::Auto,
            mtp_mode: MtpMode::Auto,
            max_seqs: MaxSeqs::Auto,
            ctx: DEFAULT_CTX,
            mem_floor_gb: DEFAULT_MEM_FLOOR_GB,
            kv_disk_dir: None,
            kv_disk_space_mb: None,
            kv_min_tokens: None,
            bank_persist_min: None,
            mtp_path: None,
            mtp_draft: None,
            sched_chunk: None,
            sched_chunk_live: None,
            native_chunk: None,
            print_plan: false,
            check_config: false,
            backend: Backend::Cuda,
            chunk_fence: ChunkFence::On,
            distribution: Distribution::Single,
            lane: LaneMode::Auto,
        }
    }
}

impl PrefixReuse {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "off" => Ok(Self::Off),
            "exact" => Ok(Self::Exact),
            "partial" => Ok(Self::Partial),
            "auto" => Ok(Self::Auto),
            _ => Err(format!(
                "ds4-server-rs: --prefix-reuse wants off|exact|partial|auto (got '{raw}')"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Exact => "exact",
            Self::Partial => "partial",
            Self::Auto => "auto",
        }
    }
}

impl MtpMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            _ => Err(format!(
                "ds4-server-rs: --mtp-mode wants off|auto|on (got '{raw}')"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Auto => "auto",
            Self::On => "on",
        }
    }
}

impl MaxSeqs {
    pub fn parse(raw: &str) -> Result<Self, String> {
        if raw == "auto" {
            return Ok(Self::Auto);
        }
        raw.parse::<u32>()
            .ok()
            .filter(|n| (1..=64).contains(n))
            .map(Self::Fixed)
            .ok_or_else(|| format!("ds4-server-rs: --max-seqs wants N|auto (got '{raw}')"))
    }

    pub fn parse_coalesce(raw: &str) -> Result<Self, String> {
        if raw == "0" {
            return Ok(Self::Off);
        }
        Self::parse(raw)
    }

    pub fn as_str(self) -> String {
        match self {
            Self::Auto => "auto".into(),
            Self::Off => "off".into(),
            Self::Fixed(n) => n.to_string(),
        }
    }
}

impl ReuseKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "off",
            Self::Exact => "exact",
            Self::Partial => "partial",
        }
    }
}

impl Support {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Present => "unverified",
            Self::Qualified => "qualified",
        }
    }
}

impl ServingRequest {
    /// Compatible aliases: `DS4_SERVER_COALESCE_MAX`, `DS4_SERVER_CONTINUOUS`,
    /// `DS4_MEM_FLOOR_GB`, `DS4_SERVER_FORK`, `DS4_SERVER_FORK_PARTIAL`,
    /// `DS4_SERVER_PERSIST_MIN_TOKENS`, chunk env vars.
    pub fn from_env() -> Self {
        let mut req = Self::default();
        if let Ok(raw) = std::env::var("DS4_SERVER_COALESCE_MAX") {
            if let Ok(parsed) = MaxSeqs::parse_coalesce(&raw) {
                req.max_seqs = parsed;
            }
        }
        // README: `DS4_SERVER_CONTINUOUS=0` forces the static/serial route.
        // It disables the continuous *lane*, not the batch context — the
        // static lane still coalesces over those banks — so it narrows the
        // driver rather than the width.
        if std::env::var_os("DS4_SERVER_CONTINUOUS").as_deref() == Some(OsStr::new("0")) {
            req.lane = LaneMode::Serial;
        }
        if let Ok(raw) = std::env::var("DS4_MEM_FLOOR_GB") {
            if let Some(gb) = parse_u64_atoi(&raw) {
                req.mem_floor_gb = gb;
            }
        }
        if std::env::var_os("DS4_CONT_PREFILL_NOFENCE").as_deref() == Some(OsStr::new("1")) {
            req.chunk_fence = ChunkFence::Off;
        }
        req.prefix_reuse = reuse_from_env();
        if std::env::var_os("DS4_MTP_SPEC_DISABLE").is_some() {
            req.mtp_mode = MtpMode::Off;
        }
        if let Ok(raw) = std::env::var("DS4_SERVER_PERSIST_MIN_TOKENS") {
            // The lane reads this through `env_i32_bound`, which clamps a
            // negative to zero and disables persistence; report that, not
            // the default this parse would otherwise fall back to.
            req.bank_persist_min = Some(
                i32::try_from(ds4_sys::libc_atoi(raw.as_bytes()))
                    .unwrap_or(i32::MAX)
                    .max(0),
            );
        }
        if let Ok(raw) = std::env::var("DS4_CONT_PREFILL_CHUNK") {
            if let Some(n) = parse_u32_atoi(&raw) {
                req.sched_chunk = Some(n);
            }
        }
        if let Ok(raw) = std::env::var("DS4_CONT_PREFILL_CHUNK_LIVE") {
            if let Some(n) = parse_u32_atoi(&raw) {
                req.sched_chunk_live = Some(n);
            }
        }
        if let Ok(raw) = std::env::var("DS4_NATIVE_PREFILL_CHUNK") {
            if let Some(n) = parse_u32_atoi(&raw) {
                req.native_chunk = Some(n);
            }
        }
        req
    }
}

pub fn parse_disk_space(raw: &str) -> Result<u64, String> {
    let trimmed = raw.trim();
    let bytes = trimmed.as_bytes();
    let (num, unit) =
        split_space(bytes).ok_or_else(|| format!("ds4-server-rs: invalid disk space '{raw}'"))?;
    let n: u64 = num
        .parse()
        .map_err(|_| format!("ds4-server-rs: invalid disk space '{raw}'"))?;
    if n == 0 {
        return Err(format!("ds4-server-rs: invalid disk space '{raw}'"));
    }
    let mb = match unit.as_str() {
        "" | "m" | "mb" | "mi" | "mib" => n,
        "g" | "gb" | "gi" | "gib" => n.saturating_mul(1024),
        "t" | "tb" | "ti" | "tib" => n.saturating_mul(1024 * 1024),
        "k" | "kb" | "ki" | "kib" => n.saturating_add(1023) / 1024,
        _ => return Err(format!("ds4-server-rs: invalid disk space unit in '{raw}'")),
    };
    // The `--kv-disk-space-mb` form is an i32. The alias must not accept a
    // budget that form rejects, and the store turns MiB into bytes.
    if mb > i32::MAX as u64 {
        return Err(format!(
            "ds4-server-rs: disk space '{raw}' exceeds {} MiB",
            i32::MAX
        ));
    }
    Ok(mb)
}

pub fn serving_caps(family: ModelFamily, variant: Variant) -> ServingCaps {
    if variant == Variant::K2Horizon375B {
        return ServingCaps {
            family,
            variant,
            banks: BankLane::Persistent,
            bank_support: Support::Qualified,
            reuse: ReuseKind::Exact,
            reuse_support: Support::Qualified,
            disk: Support::Present,
            snapshot: Support::Present,
            mtp: MtpKind::None,
            mtp_support: Support::None,
            spec_lane: SpecLane::None,
            spec_draft_min: 1,
            host: HostNeed::Graph,
            ctx_max: None,
            qualified_ctx: Some(32768),
            qualified_banks: Some(1),
            qualified_prompt: None,
            media_serial: false,
        };
    }
    match family {
        // Embedded MTP when no file is passed. A DFlash path is the external
        // draft, not those three blocks. The graph stays serial.
        ModelFamily::Mimo2 => ServingCaps {
            family,
            variant,
            banks: BankLane::Serial,
            bank_support: Support::Present,
            reuse: ReuseKind::Exact,
            reuse_support: Support::Present,
            disk: Support::Present,
            snapshot: Support::Present,
            mtp: MtpKind::Embedded,
            mtp_support: Support::Qualified,
            spec_lane: SpecLane::Serial,
            spec_draft_min: 2,
            host: HostNeed::Cuda,
            ctx_max: Some(crate::mimo2::INDEX_LIMIT),
            qualified_ctx: Some(crate::mimo2::QUALIFIED_LONG),
            qualified_banks: Some(1),
            qualified_prompt: None,
            media_serial: true,
        },
        ModelFamily::Qwen4Exp => ServingCaps {
            family,
            variant,
            banks: BankLane::Persistent,
            bank_support: Support::Qualified,
            reuse: ReuseKind::Partial,
            reuse_support: Support::Qualified,
            disk: Support::Qualified,
            snapshot: Support::Qualified,
            mtp: MtpKind::Embedded,
            mtp_support: Support::Qualified,
            spec_lane: SpecLane::Bank,
            spec_draft_min: 2,
            host: HostNeed::Cuda,
            ctx_max: Some(SHAPE_QWEN38_FLASH_NEXT.rope_orig_ctx as u32 * QWEN_YARN_MAX_FACTOR),
            qualified_ctx: Some(262144),
            qualified_banks: Some(2),
            qualified_prompt: None,
            media_serial: false,
        },
        ModelFamily::Step37 => ServingCaps {
            family,
            variant,
            banks: BankLane::OptIn,
            bank_support: Support::Qualified,
            reuse: ReuseKind::Partial,
            reuse_support: Support::Qualified,
            disk: Support::Qualified,
            snapshot: Support::Qualified,
            mtp: MtpKind::Sidecar,
            mtp_support: Support::Qualified,
            spec_lane: SpecLane::Both,
            spec_draft_min: 1,
            host: HostNeed::Cuda,
            ctx_max: Some(262144),
            qualified_ctx: Some(65536),
            qualified_banks: Some(2),
            qualified_prompt: Some(6300),
            media_serial: true,
        },
        // Ling matches the Qwen text-bank surface: two persistent banks,
        // partial reuse and disk KV. Images stay serial beside those banks
        // (Step), not on Qwen's bank-image path. No MTP predictor.
        ModelFamily::Ling3Vl => ServingCaps {
            family,
            variant,
            banks: BankLane::Persistent,
            bank_support: Support::Qualified,
            reuse: ReuseKind::Partial,
            reuse_support: Support::Qualified,
            disk: Support::Qualified,
            snapshot: Support::Qualified,
            mtp: MtpKind::None,
            mtp_support: Support::None,
            spec_lane: SpecLane::None,
            spec_draft_min: 1,
            host: HostNeed::Cuda,
            ctx_max: Some(crate::ling3vl::YARN_CONTEXT),
            qualified_ctx: Some(65536),
            qualified_banks: Some(2),
            qualified_prompt: None,
            media_serial: true,
        },
        ModelFamily::SolarOpen2 => ServingCaps {
            family,
            variant,
            banks: BankLane::Persistent,
            bank_support: Support::Qualified,
            reuse: ReuseKind::Partial,
            reuse_support: Support::Qualified,
            disk: Support::Qualified,
            snapshot: Support::Qualified,
            mtp: MtpKind::None,
            mtp_support: Support::None,
            spec_lane: SpecLane::None,
            spec_draft_min: 1,
            host: HostNeed::Graph,
            ctx_max: None,
            qualified_ctx: None,
            qualified_banks: None,
            qualified_prompt: None,
            media_serial: false,
        },
        ModelFamily::Motif3 => ServingCaps {
            family,
            variant,
            banks: BankLane::Persistent,
            bank_support: Support::Qualified,
            reuse: ReuseKind::Partial,
            reuse_support: Support::Qualified,
            disk: Support::Qualified,
            snapshot: Support::Qualified,
            mtp: MtpKind::None,
            mtp_support: Support::None,
            spec_lane: SpecLane::None,
            spec_draft_min: 1,
            host: HostNeed::Cuda,
            ctx_max: Some(262144),
            qualified_ctx: None,
            qualified_banks: None,
            qualified_prompt: None,
            media_serial: false,
        },
        ModelFamily::ExaoneMoe => ServingCaps {
            family,
            variant,
            banks: BankLane::Persistent,
            bank_support: Support::Qualified,
            reuse: ReuseKind::Partial,
            reuse_support: Support::Present,
            disk: Support::Qualified,
            snapshot: Support::Qualified,
            mtp: MtpKind::None,
            mtp_support: Support::None,
            spec_lane: SpecLane::None,
            spec_draft_min: 1,
            host: HostNeed::Graph,
            ctx_max: None,
            qualified_ctx: None,
            qualified_banks: None,
            qualified_prompt: None,
            media_serial: false,
        },
        ModelFamily::Dots3Note => ServingCaps {
            family,
            variant,
            banks: BankLane::OptIn,
            bank_support: Support::Present,
            reuse: ReuseKind::Partial,
            reuse_support: Support::Present,
            disk: Support::Present,
            snapshot: Support::Qualified,
            mtp: MtpKind::Embedded,
            mtp_support: Support::Present,
            spec_lane: SpecLane::Serial,
            spec_draft_min: 1,
            host: HostNeed::Cuda,
            ctx_max: Some(524288),
            qualified_ctx: None,
            qualified_banks: Some(1),
            qualified_prompt: None,
            media_serial: false,
        },
        ModelFamily::Inkling => ServingCaps {
            family,
            variant,
            banks: BankLane::Serial,
            bank_support: Support::None,
            reuse: ReuseKind::Exact,
            reuse_support: Support::Qualified,
            disk: Support::Present,
            snapshot: Support::Present,
            mtp: MtpKind::Sidecar,
            mtp_support: Support::Qualified,
            spec_lane: SpecLane::Serial,
            spec_draft_min: 1,
            host: HostNeed::Cuda,
            ctx_max: Some(SHAPE_INKLING_SMALL.rope_orig_ctx as u32),
            qualified_ctx: Some(1024),
            qualified_banks: Some(1),
            qualified_prompt: None,
            media_serial: false,
        },
        ModelFamily::Glm53 => ServingCaps {
            family,
            variant,
            banks: BankLane::Serial,
            bank_support: Support::None,
            reuse: ReuseKind::None,
            reuse_support: Support::None,
            disk: Support::None,
            snapshot: Support::None,
            mtp: MtpKind::None,
            mtp_support: Support::None,
            spec_lane: SpecLane::None,
            spec_draft_min: 1,
            host: HostNeed::Cuda,
            ctx_max: Some(2048),
            qualified_ctx: Some(2048),
            qualified_banks: Some(1),
            qualified_prompt: None,
            media_serial: false,
        },
        ModelFamily::DeepSeek4 => ServingCaps {
            family,
            variant,
            banks: BankLane::Persistent,
            bank_support: Support::Qualified,
            reuse: ReuseKind::Exact,
            reuse_support: Support::Qualified,
            disk: Support::Qualified,
            snapshot: Support::Qualified,
            mtp: MtpKind::DeepSeek,
            mtp_support: Support::Qualified,
            spec_lane: SpecLane::Bank,
            spec_draft_min: 1,
            host: HostNeed::Any,
            ctx_max: None,
            qualified_ctx: None,
            qualified_banks: None,
            qualified_prompt: None,
            media_serial: false,
        },
    }
}

pub fn caps_from_shape(shape: Shape) -> ServingCaps {
    serving_caps(shape.family, shape.variant)
}

pub fn caps_from_ident(id: &Identified) -> ServingCaps {
    caps_from_shape(id.shape)
}

pub fn resolve_plan(
    req: &ServingRequest,
    caps: Option<ServingCaps>,
    facts: &EngineFacts,
) -> ResolvedPlan {
    let mut issues = Vec::new();
    let requested = RequestedView {
        lane: req.lane,
        prefix_reuse: req.prefix_reuse,
        mtp_mode: req.mtp_mode,
        max_seqs: req.max_seqs,
        ctx: req.ctx,
        mem_floor_gb: req.mem_floor_gb,
        disk_dir: req.kv_disk_dir.is_some(),
        mtp_path: req.mtp_path.is_some(),
        backend: req.backend,
    };

    let Some(caps) = caps else {
        return ResolvedPlan {
            family: None,
            variant: None,
            family_name: "unknown",
            caps: None,
            requested,
            effective: default_effective(req),
            qualified: QualifiedView {
                prefix_reuse: Support::None,
                disk: Support::None,
                mtp: Support::None,
                banks: Support::None,
                ctx: None,
                banks_n: None,
                prompt: None,
                note: "no model identified; family limits unknown",
            },
            issues: vec![if req.check_config {
                error(
                    "family_unknown",
                    "--check-config requires an identified GGUF",
                )
            } else {
                warn(
                    "family_unknown",
                    "no GGUF identified; capability checks skipped",
                )
            }],
            quote: None,
        };
    };

    let (mut max_seqs, mut banks_opt_in) =
        resolve_seqs(req.max_seqs, caps, facts, req.backend, &mut issues);
    let quote = apply_quote(
        req,
        caps,
        facts,
        req.max_seqs,
        &mut max_seqs,
        &mut banks_opt_in,
        &mut issues,
    );
    let driver = bank_driver(req, caps, max_seqs, facts);
    let reuse = resolve_reuse(req, caps, driver, facts, &mut issues);
    let (mtp_mode, mtp_weights) = resolve_mtp(req, caps, facts, driver, &mut issues);
    // A draft below the family minimum allocates no speculative runtime, so
    // the plan would claim a feature that runs ordinary decode.
    let draft = req.mtp_draft.unwrap_or(caps.spec_draft_min);
    let (mtp_mode, mtp_draft) = match mtp_mode {
        MtpMode::Off => (MtpMode::Off, None),
        _ if caps.family == ModelFamily::Dots3Note && draft > DOTS3_MAX_DRAFT => {
            issues.push(error(
                "mtp_draft",
                "dots3 MTP draft exceeds the three-token trial limit",
            ));
            (MtpMode::Off, None)
        }
        mode if draft >= caps.spec_draft_min => (mode, Some(draft)),
        mode => {
            let message = format!(
                "{} speculation needs --mtp-draft of at least {}",
                caps.variant_name(),
                caps.spec_draft_min
            );
            issues.push(if mode == MtpMode::On {
                error("mtp_draft", message)
            } else {
                warn("mtp_draft", message)
            });
            (MtpMode::Off, None)
        }
    };
    let disk = resolve_disk(req, caps, facts, &mut issues);

    // The native open and session creation refuse these hosts outright, so
    // the check cannot approve the one it was pointed at.
    // No family but DeepSeek implements a distributed session, and the
    // backend each one accepts differs.
    let sliced = req.distribution == Distribution::Sliced;
    let host_refused = match caps.host {
        HostNeed::Any => false,
        HostNeed::Graph => req.backend == Backend::Cpu || sliced,
        HostNeed::Cuda => req.backend != Backend::Cuda || sliced,
    };
    if host_refused {
        issues.push(error(
            "family_host",
            match (caps.host, sliced) {
                (_, true) => format!(
                    "{} does not implement distributed layer sessions",
                    caps.variant_name()
                ),
                (HostNeed::Graph, _) => {
                    format!("{} sessions need a graph backend", caps.variant_name())
                }
                _ => format!("{} sessions need the CUDA backend", caps.variant_name()),
            },
        ));
    }

    // Both the batch context and the serial session refuse a nonpositive
    // context, so an approved plan has to have one.
    if req.ctx <= 0 {
        issues.push(error(
            "ctx_invalid",
            format!("ctx {} cannot create a session", req.ctx),
        ));
    } else if caps.ctx_max.is_some_and(|max| req.ctx as u32 > max) {
        // A hard session maximum, not a qualification bound: the server
        // would listen and then fail the first request.
        issues.push(error(
            "ctx_unavailable",
            format!(
                "{} sessions cap ctx at {}",
                caps.variant_name(),
                caps.ctx_max.unwrap_or_default()
            ),
        ));
    } else if let Some(qctx) = measured_ctx(caps, req, facts) {
        if req.ctx as u32 > qctx {
            issues.push(warn(
                "ctx_unqualified",
                format!("configured ctx {} exceeds qualified ctx {qctx}", req.ctx),
            ));
        }
    }
    if caps.qualified_prompt.is_some() {
        issues.push(warn(
            "prompt_bound",
            "configured ctx is not a full-length request proof",
        ));
    }
    if caps.media_serial && max_seqs > 1 {
        issues.push(warn(
            "media_serial",
            "image requests use the serial lane beside text banks",
        ));
    }

    if facts.ipc_pending && req.check_config {
        issues.push(error(
            "ipc_unverified",
            "check-config cannot verify IPC owner/import availability without opening the model",
        ));
    }
    if facts.artifact_ok == Some(false) {
        issues.push(error(
            "artifact_invalid",
            format!("{} artifact cannot load", caps.variant_name()),
        ));
    }
    if facts.dspark_ok == Some(false) {
        issues.push(error(
            "dspark_artifact",
            format!("{} cannot attach the DSpark drafter", caps.variant_name()),
        ));
    }
    if facts.vision_path_ok == Some(false) {
        issues.push(error(
            "vision_artifact",
            format!("{} cannot open the --vision artifact", caps.variant_name()),
        ));
    }

    let native = facts.native_chunk.or(req.native_chunk);
    let requested_boot = req.sched_chunk.unwrap_or(DEFAULT_SCHED_CHUNK);
    if req.chunk_fence == ChunkFence::On && requested_boot > PREFILL_CHUNK_FENCE {
        issues.push(warn(
            "chunk_fenced",
            format!("prefill chunk {requested_boot} is capped at {PREFILL_CHUNK_FENCE}"),
        ));
    }
    let requested_live = req.sched_chunk_live.unwrap_or(DEFAULT_SCHED_LIVE);
    if let Some(native) = native {
        if req.sched_chunk.is_some() && requested_boot > native {
            issues.push(error(
                "chunk_past_native",
                format!("prefill chunk {requested_boot} exceeds native capacity {native}"),
            ));
        }
        if req.sched_chunk_live.is_some() && requested_live > native {
            issues.push(error(
                "chunk_past_native",
                format!("live prefill chunk {requested_live} exceeds native capacity {native}"),
            ));
        }
    }
    if chunk_unverified(requested_boot, native, req.chunk_fence) {
        issues.push(error(
            "chunk_unverified",
            format!("prefill chunk {requested_boot} is below the verified set"),
        ));
    }
    if chunk_unverified(requested_live, native, req.chunk_fence) {
        issues.push(error(
            "chunk_unverified",
            format!("live prefill chunk {requested_live} is below the verified set"),
        ));
    }
    let sched_chunk = snap_verified_chunk(requested_boot, native, req.chunk_fence);
    let mut sched_live = snap_verified_chunk(requested_live, native, req.chunk_fence);
    if sched_live > sched_chunk {
        sched_live = sched_chunk;
    }

    let qualified = QualifiedView {
        // Resolution never returns a reuse stronger than the family's, and
        // exact-frontier reuse is a subset of a qualified partial path, so a
        // downgraded plan keeps the family's verification level.
        prefix_reuse: if reuse == ReuseKind::None {
            Support::None
        } else if reuse == ReuseKind::Exact && caps.variant == Variant::Kexaone236B {
            // EXAONE's existing exact path stays qualified while its new
            // checkpoint/fork path awaits its separate live gates.
            Support::Qualified
        } else {
            caps.reuse_support
        },
        disk: if disk { caps.disk } else { Support::None },
        mtp: if mtp_weights {
            caps.mtp_support
        } else {
            Support::None
        },
        banks: if max_seqs > 1 {
            caps.bank_support
        } else {
            Support::Qualified
        },
        ctx: measured_ctx(caps, req, facts),
        banks_n: caps.qualified_banks,
        prompt: caps.qualified_prompt,
        note: qualified_note(caps),
    };

    ResolvedPlan {
        family: Some(caps.family),
        variant: Some(caps.variant),
        family_name: caps.variant_name(),
        caps: Some(caps),
        requested,
        effective: EffectiveView {
            prefix_reuse: reuse,
            mtp_mode,
            mtp_weights,
            mtp_draft,
            max_seqs,
            ctx: req.ctx,
            mem_floor_gb: req.mem_floor_gb,
            disk,
            banks_opt_in,
            sched_chunk,
            sched_chunk_live: sched_live,
            bank_persist_min: req.bank_persist_min.unwrap_or(DEFAULT_BANK_PERSIST),
            disk_min_tokens: req.kv_min_tokens,
            native_chunk: native,
        },
        qualified,
        quote,
        issues,
    }
}

impl ServingCaps {
    fn variant_name(self) -> &'static str {
        match self.variant {
            Variant::Flash => "deepseek4-flash",
            Variant::Pro => "deepseek4-pro",
            Variant::SolarOpen2_250B => "solar-open2",
            Variant::Motif3 => "motif3",
            Variant::Kexaone236B => "k-exaone",
            Variant::Dots3NotePrev => "dots3-note",
            Variant::Qwen38FlashNext => "qwen4exp",
            Variant::Glm53Flash => "glm5-next",
            Variant::K2Horizon375B => "k2-horizon",
            Variant::InklingSmall => "inkling",
            Variant::Step37Flash => "step35",
            Variant::Ling30FlashVl => "bailingmoe3",
            Variant::Mimo26Flash => "mimo2",
        }
    }
}

impl ResolvedPlan {
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.level == IssueLevel::Error)
    }

    /// Listen and model-open share this gate so an impossible mix never binds.
    pub fn may_listen(&self) -> bool {
        !self.has_errors()
    }

    /// Serial-only speculation must bypass native bank fitting and routing.
    pub fn uses_serial_mtp(&self) -> bool {
        self.effective.mtp_mode != MtpMode::Off
            && self
                .caps
                .is_some_and(|caps| caps.spec_lane == SpecLane::Serial)
    }

    /// The operator asked for the bank lane by naming a width, unless the
    /// resolved MTP mode requires a serial session. Hosts must
    /// adopt this, not only the published env — a config captured before
    /// resolution keeps its own legacy `DS4_SERVER_CONTINUOUS`. `auto`
    /// expresses no preference, so it leaves that switch alone: the README
    /// promises `DS4_SERVER_CONTINUOUS=0` forces the static/serial route.
    pub fn wants_bank_lane(&self) -> bool {
        // The legacy switch and the width are orthogonal: `--max-seqs N`
        // sizes the banks the static lane coalesces over, it does not ask
        // for continuous routing the switch turned off.
        if self.uses_serial_mtp()
            || self.requested.lane == LaneMode::Serial
            || self.requested.backend != Backend::Cuda
            || self.requested.max_seqs == MaxSeqs::Off
        {
            return false;
        }
        matches!(self.requested.max_seqs, MaxSeqs::Fixed(_))
    }

    pub fn env_overrides(&self) -> Vec<(String, String)> {
        // The legacy alias round-trips: a forced-serial `0` must not come
        // back as width 1, or a re-read would re-enable the bank lane.
        let coalesce_max = match self.requested.max_seqs {
            MaxSeqs::Off => "0".to_string(),
            _ => self.effective.max_seqs.to_string(),
        };
        let mut out = vec![
            (
                "DS4_MEM_FLOOR_GB".into(),
                self.effective.mem_floor_gb.to_string(),
            ),
            ("DS4_SERVER_COALESCE_MAX".into(), coalesce_max),
        ];
        match self.effective.prefix_reuse {
            ReuseKind::None => {
                out.push(("DS4_SERVER_FORK".into(), "0".into()));
                out.push(("DS4_SERVER_FORK_PARTIAL".into(), "0".into()));
            }
            ReuseKind::Exact => {
                out.push(("DS4_SERVER_FORK".into(), "1".into()));
                out.push(("DS4_SERVER_FORK_PARTIAL".into(), "0".into()));
            }
            ReuseKind::Partial => {
                out.push(("DS4_SERVER_FORK".into(), "1".into()));
                out.push(("DS4_SERVER_FORK_PARTIAL".into(), "1".into()));
            }
        }
        if self.uses_serial_mtp() {
            out.push(("DS4_SERVER_CONTINUOUS".into(), "0".into()));
        } else if self.wants_bank_lane() {
            out.push(("DS4_SERVER_CONTINUOUS".into(), "1".into()));
        }
        // 0/1 so a later fitted-down plan can retract. Step width 1 is
        // serial MTP; Qwen width 1 is still a bank.
        if matches!(
            self.family,
            Some(ModelFamily::Step37 | ModelFamily::Dots3Note)
        ) {
            let key = if self.family == Some(ModelFamily::Dots3Note) {
                "DS4_DOTS3_BATCH"
            } else {
                "DS4_STEP37_BATCH"
            };
            out.push((
                key.into(),
                if self.effective.max_seqs > 1 {
                    "1".into()
                } else {
                    "0".into()
                },
            ));
        }
        if self.family == Some(ModelFamily::Qwen4Exp) {
            out.push((
                "DS4_QWEN_BATCH".into(),
                if self.requested.backend == Backend::Cuda
                    && self.effective.max_seqs >= 1
                    && self.requested.max_seqs != MaxSeqs::Off
                {
                    "1".into()
                } else {
                    "0".into()
                },
            ));
        }
        if self.effective.mtp_mode == MtpMode::Off {
            out.push(("DS4_MTP_SPEC_DISABLE".into(), "1".into()));
        }
        if self.family == Some(ModelFamily::Dots3Note) {
            out.push((
                "DS4_DOTS3_MTP".into(),
                if self.effective.mtp_mode == MtpMode::On {
                    "1"
                } else {
                    "0"
                }
                .into(),
            ));
        }
        // Never publish a yield past the allocated native workspace.
        let boot = published_chunk(self.effective.sched_chunk, self.effective.native_chunk);
        let live = published_chunk(self.effective.sched_chunk_live, self.effective.native_chunk);
        out.push(("DS4_CONT_PREFILL_CHUNK".into(), boot.to_string()));
        out.push(("DS4_CONT_PREFILL_CHUNK_LIVE".into(), live.to_string()));
        // C family allocators read DS4_*_PREFILL_CHUNK, not --native-chunk.
        if let Some(native) = self.effective.native_chunk {
            if let Some(key) = self.native_prefill_env() {
                out.push((key.into(), native.to_string()));
            }
        }
        out
    }

    fn native_prefill_env(&self) -> Option<&'static str> {
        match self.family {
            Some(ModelFamily::Qwen4Exp) => Some("DS4_QWEN_PREFILL_CHUNK"),
            Some(ModelFamily::Step37) => Some("DS4_STEP37_PREFILL_CHUNK"),
            Some(ModelFamily::Ling3Vl) => Some("DS4_LING3VL_PREFILL_CHUNK"),
            Some(ModelFamily::Inkling) => Some("DS4_INKLING_PREFILL_CHUNK"),
            Some(ModelFamily::ExaoneMoe) => Some("DS4_EXAONE_PREFILL_CHUNK"),
            Some(ModelFamily::Motif3) => Some("DS4_MOTIF3_PREFILL_CHUNK"),
            Some(ModelFamily::SolarOpen2 | ModelFamily::DeepSeek4) => {
                Some("DS4_METAL_PREFILL_CHUNK")
            }
            Some(ModelFamily::Dots3Note) => Some("DS4_DOTS3_PREFILL_CHUNK"),
            Some(ModelFamily::Mimo2) => Some("DS4_MIMO2_PREFILL_CHUNK"),
            _ => None,
        }
    }

    fn controls_json(&self) -> Value {
        // Proposals use the same allocator mapping and capability table as
        // admission. An unidentified model or workspace offers no chunk choices.
        let scheduler_chunks: Vec<_> = self
            .caps
            .and(self.effective.native_chunk)
            .map(|cap| {
                VERIFIED_PREFILL_CHUNKS
                    .into_iter()
                    .filter(|n| *n <= cap)
                    .collect()
            })
            .unwrap_or_default();
        json!({
            "native_prefill_env": self.caps.and_then(|_| self.native_prefill_env()),
            "scheduler_chunks": scheduler_chunks,
            "prefix_reuse": self.caps.map(|caps| match caps.reuse {
                ReuseKind::None => "none",
                ReuseKind::Exact => "exact",
                ReuseKind::Partial => "partial",
            }),
            "banks": self.caps.map(|caps| match caps.banks {
                BankLane::Serial => "serial",
                BankLane::OptIn => "opt_in",
                BankLane::Persistent => "persistent",
            }),
            "mtp": self.caps.map(|caps| caps.mtp_support.as_str()),
            "disk": self.caps.map(|caps| caps.disk.as_str()),
        })
    }

    /// Pass the quoted workspace to native allocators that consume this argument.
    pub fn batch_max_total_tokens(&self, ctx: i32, width: i32) -> i32 {
        match self.family {
            Some(ModelFamily::DeepSeek4) => {
                self.effective
                    .native_chunk
                    .unwrap_or(DEFAULT_SCHED_CHUNK.min(ctx.max(1) as u32)) as i32
            }
            Some(ModelFamily::SolarOpen2) => {
                self.effective.native_chunk.map(|n| n as i32).unwrap_or(0)
            }
            _ => ctx.saturating_mul(width),
        }
    }

    pub fn apply_env(&self) {
        if self.effective.mtp_mode != MtpMode::Off {
            std::env::remove_var("DS4_MTP_SPEC_DISABLE");
        }
        for (key, value) in self.env_overrides() {
            std::env::set_var(key, value);
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "family": self.family_name,
            "controls": self.controls_json(),
            "requested": {
                "prefix_reuse": self.requested.prefix_reuse.as_str(),
                "mtp_mode": self.requested.mtp_mode.as_str(),
                "max_seqs": self.requested.max_seqs.as_str(),
                "lane": match self.requested.lane {
                    LaneMode::Auto => "auto",
                    LaneMode::Serial => "serial",
                },
                "ctx": self.requested.ctx,
                "mem_floor_gb": self.requested.mem_floor_gb,
                "disk": self.requested.disk_dir,
                "mtp_path": self.requested.mtp_path,
                "backend": backend_name(self.requested.backend)
            },
            "effective": {
                "prefix_reuse": self.effective.prefix_reuse.as_str(),
                "mtp_mode": self.effective.mtp_mode.as_str(),
                "mtp_weights": self.effective.mtp_weights,
                "mtp_draft": self.effective.mtp_draft,
                "max_seqs": self.effective.max_seqs,
                "ctx": self.effective.ctx,
                "mem_floor_gb": self.effective.mem_floor_gb,
                "disk": self.effective.disk,
                "banks_opt_in": self.effective.banks_opt_in,
                "sched_chunk": self.effective.sched_chunk,
                "sched_chunk_live": self.effective.sched_chunk_live,
                "native_chunk": self.effective.native_chunk,
                "bank_persist_min_tokens": self.effective.bank_persist_min,
                "disk_min_tokens": self.effective.disk_min_tokens,
                "disk_is_offload": false
            },
            "quote": self.quote.map(|q| json!({
                "shared_weights": q.shared_weights,
                "per_bank": q.per_bank,
                "mtp_state": q.mtp_state,
                "scratch": q.scratch,
                "checkpoint_pool": q.checkpoint_pool,
                "ple": q.ple,
                "media_reserve": q.media_reserve,
                "floor": q.floor,
                "available": q.available,
                "banks": q.banks,
                "total": q.total
            })),
            "qualified": {
                "prefix_reuse": self.qualified.prefix_reuse.as_str(),
                "disk": self.qualified.disk.as_str(),
                "mtp": self.qualified.mtp.as_str(),
                "banks": self.qualified.banks.as_str(),
                "ctx": self.qualified.ctx,
                "banks_n": self.qualified.banks_n,
                "prompt": self.qualified.prompt,
                "note": self.qualified.note
            },
            "issues": self.issues.iter().map(|i| json!({
                "level": match i.level {
                    IssueLevel::Error => "error",
                    IssueLevel::Warn => "warn",
                },
                "code": i.code,
                "message": i.message
            })).collect::<Vec<_>>()
        })
    }

    pub fn report(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "family: {}", self.family_name);
        let _ = writeln!(
            s,
            "requested: reuse={} max_seqs={} mtp={} ctx={} floor={}G disk={}",
            self.requested.prefix_reuse.as_str(),
            self.requested.max_seqs.as_str(),
            self.requested.mtp_mode.as_str(),
            self.requested.ctx,
            self.requested.mem_floor_gb,
            self.requested.disk_dir
        );
        let _ = writeln!(
            s,
            "effective: reuse={} max_seqs={} mtp={} weights={} ctx={} floor={}G disk={}",
            self.effective.prefix_reuse.as_str(),
            self.effective.max_seqs,
            self.effective.mtp_mode.as_str(),
            self.effective.mtp_weights,
            self.effective.ctx,
            self.effective.mem_floor_gb,
            self.effective.disk
        );
        let _ = writeln!(
            s,
            "qualified: reuse={} disk={} mtp={} banks={} ctx={:?} prompt={:?}",
            self.qualified.prefix_reuse.as_str(),
            self.qualified.disk.as_str(),
            self.qualified.mtp.as_str(),
            self.qualified.banks.as_str(),
            self.qualified.ctx,
            self.qualified.prompt
        );
        if !self.qualified.note.is_empty() {
            let _ = writeln!(s, "note: {}", self.qualified.note);
        }
        for issue in &self.issues {
            let _ = writeln!(
                s,
                "{}: {} ({})",
                issue_tag(issue.level),
                issue.message,
                issue.code
            );
        }
        s
    }
}

impl ReuseTaken {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Exact => "exact",
            Self::Partial => "partial",
            Self::Fork => "fork",
        }
    }
}

impl RequestTrace {
    pub fn to_json(&self) -> Value {
        let mut trace = json!({
            "effective_lane": self.effective_lane,
            "reuse_kind": self.reuse_kind.as_str(),
            "speculation_active": self.speculation_active,
            "fallback_reason": self.fallback_reason
        });
        // Absent when nothing was refused, as the contract says. A null
        // member would read as "there is a miss, and it has no reason",
        // and a client testing for the key would believe it.
        if self.reuse_miss != ReuseMiss::None {
            trace["reuse_miss"] = json!(self.reuse_miss.as_str());
        }
        trace
    }
}

fn default_effective(req: &ServingRequest) -> EffectiveView {
    EffectiveView {
        prefix_reuse: ReuseKind::None,
        mtp_mode: req.mtp_mode,
        mtp_weights: req.mtp_path.is_some(),
        mtp_draft: None,
        max_seqs: match req.max_seqs {
            MaxSeqs::Auto | MaxSeqs::Off => 1,
            MaxSeqs::Fixed(n) => n,
        },
        ctx: req.ctx,
        mem_floor_gb: req.mem_floor_gb,
        disk: req.kv_disk_dir.is_some(),
        banks_opt_in: false,
        sched_chunk: req.sched_chunk.unwrap_or(DEFAULT_SCHED_CHUNK),
        sched_chunk_live: req.sched_chunk_live.unwrap_or(DEFAULT_SCHED_LIVE),
        bank_persist_min: req.bank_persist_min.unwrap_or(DEFAULT_BANK_PERSIST),
        disk_min_tokens: req.kv_min_tokens,
        native_chunk: req.native_chunk,
    }
}

fn resolve_reuse(
    req: &ServingRequest,
    caps: ServingCaps,
    driver: BankDriver,
    facts: &EngineFacts,
    issues: &mut Vec<PlanIssue>,
) -> ReuseKind {
    match req.prefix_reuse {
        PrefixReuse::Off => ReuseKind::None,
        PrefixReuse::Exact => {
            if caps.reuse == ReuseKind::None {
                issues.push(error(
                    "reuse_unsupported",
                    format!("{} has no prefix reuse", caps.variant_name()),
                ));
                ReuseKind::None
            } else {
                ReuseKind::Exact
            }
        }
        PrefixReuse::Partial => {
            if caps.reuse != ReuseKind::Partial {
                issues.push(error(
                    "partial_unsupported",
                    format!(
                        "{} does not provide partial reuse (best qualified: {})",
                        caps.variant_name(),
                        caps.reuse.as_str()
                    ),
                ));
                return caps.reuse;
            }
            match partial_block(driver, facts) {
                Some(block) => {
                    issues.push(error(block.code(), block.message(caps)));
                    ReuseKind::Exact
                }
                None => ReuseKind::Partial,
            }
        }
        PrefixReuse::Auto => {
            if caps.reuse != ReuseKind::Partial {
                return caps.reuse;
            }
            if caps.reuse_support != Support::Qualified {
                issues.push(warn(
                    "partial_unqualified",
                    format!(
                        "{} partial reuse is implemented but not qualified",
                        caps.variant_name()
                    ),
                ));
                return ReuseKind::Exact;
            }
            match partial_block(driver, facts) {
                Some(block) => {
                    issues.push(warn(block.code(), block.message(caps)));
                    ReuseKind::Exact
                }
                None => ReuseKind::Partial,
            }
        }
    }
}

/// Why partial reuse cannot run in this process. Checkpoint replay lives in
/// the bank driver and needs the runtime's checkpoint store; the serial
/// path only ever extends an exact prefix.
#[derive(Clone, Copy)]
enum PartialBlock {
    Lane,
    Runtime,
}

impl PartialBlock {
    fn code(self) -> &'static str {
        match self {
            Self::Lane => "partial_lane",
            Self::Runtime => "partial_runtime",
        }
    }

    fn message(self, caps: ServingCaps) -> String {
        match self {
            Self::Lane => format!(
                "{} partial reuse runs in the bank lane; this plan has none",
                caps.variant_name()
            ),
            Self::Runtime => format!(
                "{} runtime opened without a partial checkpoint store",
                caps.variant_name()
            ),
        }
    }
}

fn partial_block(driver: BankDriver, facts: &EngineFacts) -> Option<PartialBlock> {
    if driver == BankDriver::Absent {
        return Some(PartialBlock::Lane);
    }
    (facts.partial_reuse == Some(false)).then_some(PartialBlock::Runtime)
}

fn chunk_cap(want: u32, native: Option<u32>, fence: ChunkFence) -> u32 {
    let mut cap = want;
    if fence == ChunkFence::On {
        cap = cap.min(PREFILL_CHUNK_FENCE);
    }
    if let Some(native) = native {
        cap = cap.min(native);
    }
    cap
}

fn chunk_unverified(want: u32, native: Option<u32>, fence: ChunkFence) -> bool {
    if want == 0 || fence == ChunkFence::Off {
        return false;
    }
    let cap = chunk_cap(want, native, fence);
    // A native cap below the verified scheduler widths is a short tail.
    // An explicitly smaller yield still needs the normal verification fence.
    if native == Some(cap) && cap > 0 && cap < VERIFIED_PREFILL_CHUNKS[0] {
        return false;
    }
    !VERIFIED_PREFILL_CHUNKS.iter().any(|n| *n <= cap)
}

fn snap_verified_chunk(want: u32, native: Option<u32>, fence: ChunkFence) -> u32 {
    if want == 0 {
        return 0;
    }
    let cap = chunk_cap(want, native, fence);
    if fence == ChunkFence::Off
        || (native == Some(cap) && cap > 0 && cap < VERIFIED_PREFILL_CHUNKS[0])
    {
        return cap;
    }
    VERIFIED_PREFILL_CHUNKS
        .iter()
        .rev()
        .copied()
        .find(|n| *n <= cap)
        .unwrap_or(VERIFIED_PREFILL_CHUNKS[0])
}

fn published_chunk(n: u32, native: Option<u32>) -> u32 {
    native.map_or(n, |cap| n.min(cap))
}

fn apply_quote(
    req: &ServingRequest,
    caps: ServingCaps,
    facts: &EngineFacts,
    requested: MaxSeqs,
    max_seqs: &mut u32,
    banks_opt_in: &mut bool,
    issues: &mut Vec<PlanIssue>,
) -> Option<ServingQuote> {
    facts.host_available_bytes?;
    match quoted_width(*max_seqs, req, facts) {
        Some(n) if n < *max_seqs => {
            if matches!(requested, MaxSeqs::Fixed(_)) {
                issues.push(error(
                    "banks_not_quoted",
                    format!(
                        "requested {} banks but the memory quote fits {n}",
                        *max_seqs
                    ),
                ));
            }
            *max_seqs = n;
            *banks_opt_in = n > 1 && caps.banks == BankLane::OptIn;
        }
        None => {
            issues.push(error(
                "quote_overflow",
                "memory quote cannot host the mix at one bank",
            ));
            *max_seqs = 1;
            *banks_opt_in = false;
        }
        Some(_) => {}
    }
    serving_quote(req, facts, *max_seqs)
}

fn serving_quote(req: &ServingRequest, facts: &EngineFacts, banks: u32) -> Option<ServingQuote> {
    let available = facts.host_available_bytes?;
    let mut quote = ServingQuote {
        shared_weights: facts.shared_weights_bytes.unwrap_or(0),
        per_bank: facts.per_bank_bytes.unwrap_or(0),
        mtp_state: facts.mtp_state_bytes.unwrap_or(0),
        scratch: facts.scratch_bytes.unwrap_or(0),
        checkpoint_pool: facts.checkpoint_pool_bytes.unwrap_or(0),
        ple: facts.ple_bytes.unwrap_or(0),
        media_reserve: facts.media_reserve_bytes.unwrap_or(0).saturating_add(
            facts
                .media_per_extra_bank_bytes
                .unwrap_or(0)
                .saturating_mul(u64::from(banks.saturating_sub(1))),
        ),
        floor: req
            .mem_floor_gb
            .saturating_mul(GIB)
            .max(facts.fit_headroom_bytes.unwrap_or(0)),
        available,
        banks,
        total: 0,
    };
    quote.total = quote.cost(banks);
    Some(quote)
}

fn quoted_width(want: u32, req: &ServingRequest, facts: &EngineFacts) -> Option<u32> {
    let mut n = want.max(1);
    while n >= 1 {
        // Reprice lazy per-bank reserves for each candidate, including width 1.
        let quote = serving_quote(req, facts, n)?;
        if quote.total <= quote.available {
            return Some(n);
        }
        if n == 1 {
            break;
        }
        n -= 1;
    }
    None
}

fn resolve_seqs(
    requested: MaxSeqs,
    caps: ServingCaps,
    facts: &EngineFacts,
    backend: Backend,
    issues: &mut Vec<PlanIssue>,
) -> (u32, bool) {
    // Auto: serial/Step stay 1 so `-m` boots. Persistent uses qualified
    // banks. Only explicit `--max-seqs N>1` errors on serial.
    let mut want = match requested {
        MaxSeqs::Off => 1,
        MaxSeqs::Auto => match caps.banks {
            BankLane::Serial | BankLane::OptIn => 1,
            BankLane::Persistent => caps.qualified_banks.unwrap_or(DEFAULT_MAX_SEQS),
        },
        MaxSeqs::Fixed(n) => n,
    };
    // Two text banks that 503 every image request are not a mixed-modal
    // default. A quoted serial-media reserve can still admit more than one.
    if requested == MaxSeqs::Auto && caps.media_serial && facts.media_reserve_bytes.is_none() {
        want = want.min(1);
    }
    if requested == MaxSeqs::Off {
        return (1, false);
    }
    // Continuous banks are CUDA-only. CPU/Metal Auto is width 1.
    if backend != Backend::Cuda {
        if let MaxSeqs::Fixed(n) = requested {
            if n > 1 {
                issues.push(error(
                    "banks_cuda",
                    format!(
                        "{} banks need CUDA; --backend {} cannot run --max-seqs {n}",
                        caps.variant_name(),
                        backend_name(backend)
                    ),
                ));
            }
        }
        return (1, false);
    }
    if let MaxSeqs::Fixed(n) = requested {
        if n > 1 && caps.banks == BankLane::Serial {
            issues.push(error(
                "banks_unsupported",
                format!(
                    "{} live serving is serial; --max-seqs {n} is not available",
                    caps.variant_name()
                ),
            ));
            return (1, false);
        }
    }
    let fitted = facts.banks_fitted.unwrap_or(want);
    let n = fitted.min(want);
    // A forced width that the fit reduces is a silently narrower deployment.
    // Auto may shrink; `--max-seqs N` may not.
    if matches!(requested, MaxSeqs::Fixed(_)) && n < want {
        issues.push(error(
            "banks_not_fitted",
            format!("requested {want} banks but native fitted {n}"),
        ));
    }
    if let Some(qb) = caps.qualified_banks {
        if n > qb {
            issues.push(warn(
                "banks_unqualified",
                format!("max_seqs {n} exceeds qualified banks {qb}"),
            ));
        }
    }
    if caps.family == ModelFamily::Mimo2 {
        issues.push(warn(
            "seqs_omitted",
            "max_seqs 2 is omitted; the MiMo graph is serial",
        ));
    }
    let opt_in = n > 1 && caps.banks == BankLane::OptIn;
    (n, opt_in)
}

fn resolve_mtp(
    req: &ServingRequest,
    caps: ServingCaps,
    facts: &EngineFacts,
    driver: BankDriver,
    issues: &mut Vec<PlanIssue>,
) -> (MtpMode, bool) {
    let has_path = req.mtp_path.is_some();
    let mimo_dflash = caps.family == ModelFamily::Mimo2 && has_path;
    let can = match caps.mtp {
        MtpKind::None | MtpKind::BoundOnly => false,
        MtpKind::Embedded | MtpKind::Sidecar | MtpKind::DeepSeek => true,
    };
    if req.backend != Backend::Cuda && facts.mtp_path_ok != Some(false) {
        if req.mtp_mode == MtpMode::On {
            issues.push(error(
                "mtp_cuda",
                format!(
                    "{} MTP needs CUDA; --backend {} cannot enable it",
                    caps.variant_name(),
                    backend_name(req.backend)
                ),
            ));
        }
        return (MtpMode::Off, false);
    }
    // Embedded predictors use the main artifact; only sidecar families
    // take a separate path.
    if has_path && !mimo_dflash && !matches!(caps.mtp, MtpKind::Sidecar | MtpKind::DeepSeek) {
        issues.push(error(
            "mtp_contract",
            format!(
                "{} does not take an MTP sidecar; drop --mtp",
                caps.variant_name()
            ),
        ));
        return (MtpMode::Off, false);
    }
    // A named artifact the host cannot attach breaks every mode: it still
    // goes to `Model::open_*`, so `off` and `auto` fail at boot too.
    if has_path && facts.mtp_path_ok == Some(false) {
        issues.push(error(
            "mtp_sidecar",
            format!(
                "{} MTP sidecar is missing, is not a GGUF, or does not attach",
                caps.variant_name()
            ),
        ));
        return (MtpMode::Off, false);
    }
    // Qwen/DeepSeek speculation lives in the bank driver. The legacy zero
    // alias (or a refused fit) routes every request through NativeDecode,
    // which only speculates for Inkling and Step, so MTP would never run.
    if caps.spec_lane == SpecLane::Bank && driver == BankDriver::Absent {
        if req.mtp_mode == MtpMode::On {
            issues.push(error(
                "mtp_lane",
                format!(
                    "{} MTP runs on the continuous lane; serial serving cannot enable it",
                    caps.variant_name()
                ),
            ));
        }
        return (MtpMode::Off, false);
    }
    if caps.spec_lane == SpecLane::Serial && driver == BankDriver::Present {
        if req.mtp_mode == MtpMode::On {
            issues.push(error(
                "mtp_lane",
                format!("{} MTP runs only on the serial lane", caps.variant_name()),
            ));
        }
        return (MtpMode::Off, false);
    }
    if req.mtp_mode == MtpMode::On && caps.mtp == MtpKind::BoundOnly {
        issues.push(error(
            "mtp_unexecuted",
            format!(
                "{} binds MTP weights but does not execute them",
                caps.variant_name()
            ),
        ));
        return (MtpMode::Off, false);
    }
    if req.mtp_mode == MtpMode::On && !can && !facts.mtp_loaded && !has_path {
        issues.push(error(
            "mtp_unsupported",
            format!("{} cannot run native MTP", caps.variant_name()),
        ));
        return (MtpMode::Off, false);
    }
    if req.mtp_mode == MtpMode::On
        && matches!(caps.mtp, MtpKind::Sidecar | MtpKind::DeepSeek)
        && !has_path
        && !facts.mtp_loaded
    {
        issues.push(error(
            "mtp_sidecar",
            format!("{} MTP on requires --mtp PATH", caps.variant_name()),
        ));
        return (MtpMode::Off, false);
    }
    let weights = match caps.mtp {
        MtpKind::Embedded => {
            req.mtp_mode != MtpMode::Off
                && (mimo_dflash
                    || req.mtp_mode == MtpMode::On
                    || (req.mtp_mode == MtpMode::Auto && caps.mtp_support == Support::Qualified))
        }
        MtpKind::Sidecar | MtpKind::DeepSeek => has_path || facts.mtp_loaded,
        MtpKind::BoundOnly | MtpKind::None => false,
    };
    let mode = match req.mtp_mode {
        MtpMode::Off => MtpMode::Off,
        MtpMode::On if weights => MtpMode::On,
        MtpMode::On => MtpMode::Off,
        MtpMode::Auto if caps.mtp_support != Support::Qualified => MtpMode::Off,
        MtpMode::Auto if weights => MtpMode::Auto,
        MtpMode::Auto => MtpMode::Off,
    };
    if mode == MtpMode::On && caps.mtp_support == Support::Present {
        issues.push(warn(
            "mtp_unverified",
            "MTP execution is present but not qualified",
        ));
    }
    (mode, weights)
}

/// Whether this plan will have a bank driver at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankDriver {
    Present,
    Absent,
}

/// Absent when the operator forced serial through either legacy switch, the
/// backend has no lane, the native fit refused it, the family serves
/// serially, or an opt-in family stayed at width one.
///
/// This mirrors the native admission gate: Inkling and GLM refuse banks
/// outright; Qwen, Step and dots3 require their batch environment switch to
/// be `1` (which `env_overrides` publishes from the resolved width), and
/// the remaining families are persistent.
fn bank_driver(
    req: &ServingRequest,
    caps: ServingCaps,
    width: u32,
    facts: &EngineFacts,
) -> BankDriver {
    let absent = req.max_seqs == MaxSeqs::Off
        || req.lane == LaneMode::Serial
        || req.backend != Backend::Cuda
        || facts.cont_lane == Some(false)
        || caps.banks == BankLane::Serial
        || (caps.banks == BankLane::OptIn && width < 2);
    if absent {
        BankDriver::Absent
    } else {
        BankDriver::Present
    }
}

fn resolve_disk(
    req: &ServingRequest,
    caps: ServingCaps,
    facts: &EngineFacts,
    issues: &mut Vec<PlanIssue>,
) -> bool {
    let want = req.kv_disk_dir.is_some();
    if !want {
        return false;
    }
    if caps.disk == Support::None || caps.snapshot == Support::None {
        issues.push(error(
            "disk_unsupported",
            format!("{} session snapshots are unsupported", caps.variant_name()),
        ));
        return false;
    }
    if facts.disk_ready == Some(false) {
        issues.push(error("disk_open", "KV disk store could not be opened"));
        return false;
    }
    if caps.disk == Support::Present {
        issues.push(warn(
            "disk_unverified",
            format!(
                "{} disk KV is present but not qualified",
                caps.variant_name()
            ),
        ));
    }
    true
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda => "cuda",
        Backend::Cpu => "cpu",
        Backend::Metal => "metal",
    }
}

/// 512k text was measured without the projector and without DFlash.
/// The two together were measured at 262144, so that plan must not
/// publish 524288 as the qualified context.
/// `--mtp-mode off` must not open a draft width. MiMo speculation turns on
/// from that width alone, so a leftover `--mtp-draft` would ignore the mode.
pub fn open_draft_tokens(
    mode: MtpMode,
    requested: Option<i32>,
    planned: Option<i32>,
) -> Option<i32> {
    if mode == MtpMode::Off {
        return None;
    }
    requested.filter(|n| *n > 0).or(planned.filter(|n| *n > 0))
}

fn measured_ctx(caps: ServingCaps, req: &ServingRequest, facts: &EngineFacts) -> Option<u32> {
    if caps.family == ModelFamily::Mimo2
        && req.mtp_path.is_some()
        && (facts.vision_loaded || facts.vision_path_ok == Some(true))
    {
        return Some(crate::mimo2::QUALIFIED_CONTEXT);
    }
    caps.qualified_ctx
}

fn qualified_note(caps: ServingCaps) -> &'static str {
    match caps.variant {
        Variant::Step37Flash => {
            "text banks are opt-in; Chat restart hits need history-stable identity; images serial"
        }
        Variant::K2Horizon375B => {
            "32K one-bank serving is qualified; disk KV and external owner import are not"
        }
        Variant::Glm53Flash => "serial graph is capped at 2,048 tokens; snapshots unsupported",
        Variant::Dots3NotePrev => {
            "text banks, local-window partial reuse and serial MTP are present but unqualified"
        }
        Variant::InklingSmall => "serial text snapshots present; media snapshots unsupported",
        Variant::Kexaone236B => {
            "exact reuse qualified; LLLG partial checkpoints await live qualification"
        }
        Variant::Qwen38FlashNext => {
            "common UX baseline; configured values and verified combinations differ"
        }
        Variant::Mimo26Flash => {
            "512k serial text is the qualified context. Embedded MTP is qualified when no DFlash file is loaded. DFlash, the projector, image, audio, and video are qualified together at context 262144 and prefill chunk 4096. 512k with the projector and DFlash was not measured. max_seqs 2 is omitted because the graph is serial. 1M is not qualified"
        }
        _ => "",
    }
}

fn reuse_from_env() -> PrefixReuse {
    let fork = std::env::var("DS4_SERVER_FORK").ok();
    let partial = std::env::var("DS4_SERVER_FORK_PARTIAL").ok();
    if fork.as_deref() == Some("0") {
        return PrefixReuse::Off;
    }
    if partial.as_deref() == Some("0") {
        return PrefixReuse::Exact;
    }
    if partial.as_deref() == Some("1") {
        return PrefixReuse::Partial;
    }
    PrefixReuse::Auto
}

fn split_space(raw: &[u8]) -> Option<(String, String)> {
    let mut end = raw.len();
    while end > 0 && raw[end - 1].is_ascii_alphabetic() {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let num = std::str::from_utf8(&raw[..end]).ok()?.trim();
    let unit = std::str::from_utf8(&raw[end..])
        .ok()?
        .trim()
        .to_ascii_lowercase();
    Some((num.to_string(), unit))
}

fn parse_u64_atoi(raw: &str) -> Option<u64> {
    let v = ds4_sys::libc_atoi(raw.as_bytes());
    if v < 0 {
        None
    } else {
        Some(v as u64)
    }
}

fn parse_u32_atoi(raw: &str) -> Option<u32> {
    parse_u64_atoi(raw).and_then(|n| u32::try_from(n).ok())
}

fn error(code: &'static str, message: impl Into<String>) -> PlanIssue {
    PlanIssue {
        level: IssueLevel::Error,
        code,
        message: message.into(),
    }
}

fn warn(code: &'static str, message: impl Into<String>) -> PlanIssue {
    PlanIssue {
        level: IssueLevel::Warn,
        code,
        message: message.into(),
    }
}

fn issue_tag(level: IssueLevel) -> &'static str {
    match level {
        IssueLevel::Error => "error",
        IssueLevel::Warn => "warn",
    }
}

impl fmt::Display for ResolvedPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_trace_omits_the_miss() {
        let mut trace = RequestTrace {
            effective_lane: "serial",
            reuse_kind: ReuseTaken::Exact,
            reuse_miss: ReuseMiss::None,
            speculation_active: false,
            fallback_reason: None,
        };
        let json = trace.to_json();
        assert!(json.get("reuse_miss").is_none(), "{json}");
        assert_eq!(json["reuse_kind"], "exact");

        trace.reuse_miss = ReuseMiss::BelowThreshold;
        assert_eq!(
            trace.to_json()["reuse_miss"],
            "below minimum token threshold"
        );
    }

    fn caps(family: ModelFamily, variant: Variant) -> ServingCaps {
        serving_caps(family, variant)
    }

    fn plan(req: ServingRequest, family: ModelFamily, variant: Variant) -> ResolvedPlan {
        resolve_plan(&req, Some(caps(family, variant)), &EngineFacts::default())
    }

    #[test]
    fn check_requires_identified_model() {
        let req = ServingRequest {
            check_config: true,
            ..ServingRequest::default()
        };
        let p = resolve_plan(&req, None, &EngineFacts::default());
        assert!(p.has_errors());
    }

    #[test]
    fn controls_follow_family_caps() {
        let req = ServingRequest {
            native_chunk: Some(1280),
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::ExaoneMoe, Variant::K2Horizon375B);
        let controls = p.to_json()["controls"].clone();
        assert_eq!(controls["native_prefill_env"], "DS4_EXAONE_PREFILL_CHUNK");
        assert_eq!(controls["scheduler_chunks"], json!([256, 512, 1024]));
        assert_eq!(controls["prefix_reuse"], "exact");
        assert_eq!(controls["banks"], "persistent");
        assert_eq!(controls["mtp"], "none");
        assert_eq!(controls["disk"], "unverified");
        assert!(p.env_overrides().iter().any(|(key, value)| {
            key == controls["native_prefill_env"].as_str().unwrap() && value == "1280"
        }));

        let req = ServingRequest {
            ctx: 2048,
            ..ServingRequest::default()
        };
        let glm = plan(req, ModelFamily::Glm53, Variant::Glm53Flash).to_json();
        assert!(glm["controls"]["native_prefill_env"].is_null());
        assert_eq!(glm["controls"]["prefix_reuse"], "none");
        assert_eq!(glm["controls"]["banks"], "serial");
        assert_eq!(glm["controls"]["disk"], "none");
    }

    #[test]
    fn controls_need_known_capacity() {
        let p = plan(
            ServingRequest::default(),
            ModelFamily::ExaoneMoe,
            Variant::K2Horizon375B,
        );
        assert_eq!(p.to_json()["controls"]["scheduler_chunks"], json!([]));
        let unknown = resolve_plan(
            &ServingRequest {
                native_chunk: Some(2048),
                ..ServingRequest::default()
            },
            None,
            &EngineFacts::default(),
        );
        let controls = unknown.to_json()["controls"].clone();
        assert_eq!(controls["scheduler_chunks"], json!([]));
        for key in ["native_prefill_env", "prefix_reuse", "banks", "mtp", "disk"] {
            assert!(controls[key].is_null(), "{key}: {controls}");
        }
    }

    #[test]
    fn off_mode_drops_a_requested_draft_width() {
        assert_eq!(open_draft_tokens(MtpMode::Off, Some(8), Some(2)), None);
        assert_eq!(open_draft_tokens(MtpMode::On, Some(8), Some(2)), Some(8));
        assert_eq!(open_draft_tokens(MtpMode::Auto, None, Some(2)), Some(2));
    }

    #[test]
    fn mtp_on_does_not_publish_disable() {
        // Qwen's MTP is embedded; a sidecar path would be a contract error.
        let req = ServingRequest {
            mtp_mode: MtpMode::On,
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p.has_errors());
        assert_eq!(p.effective.mtp_mode, MtpMode::On);
        assert!(!p
            .env_overrides()
            .iter()
            .any(|(key, _)| key == "DS4_MTP_SPEC_DISABLE"));
    }

    #[test]
    fn an_embedded_mtp_family_takes_no_sidecar() {
        let mut req = ServingRequest::default();
        req.mtp_path = Some("mtp.gguf".into());
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "mtp_contract"));
    }

    #[test]
    fn mtp_off_disables_embedded() {
        let req = ServingRequest {
            mtp_mode: MtpMode::Off,
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p
            .env_overrides()
            .iter()
            .any(|(key, value)| key == "DS4_MTP_SPEC_DISABLE" && value == "1"));
    }

    #[test]
    fn forced_banks_below_the_fit_are_an_error() {
        let req = ServingRequest {
            max_seqs: MaxSeqs::Fixed(2),
            ..ServingRequest::default()
        };
        let facts = EngineFacts {
            banks_fitted: Some(1),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &req,
            Some(caps(ModelFamily::Step37, Variant::Step37Flash)),
            &facts,
        );
        assert_eq!(p.requested.max_seqs, MaxSeqs::Fixed(2));
        assert_eq!(p.effective.max_seqs, 1);
        assert!(p.has_errors());
        assert!(p
            .issues
            .iter()
            .any(|issue| issue.code == "banks_not_fitted"));
        assert!(p
            .env_overrides()
            .iter()
            .any(|(key, value)| key == "DS4_STEP37_BATCH" && value == "0"));
    }

    #[test]
    fn auto_banks_below_the_fit_stay_a_warning() {
        let facts = EngineFacts {
            banks_fitted: Some(1),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &ServingRequest::default(),
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert_eq!(p.effective.max_seqs, 1);
        assert!(!p.has_errors());
    }

    #[test]
    fn an_oversized_chunk_resolves_to_the_fence() {
        let mut req = ServingRequest::default();
        req.sched_chunk = Some(16384);
        req.sched_chunk_live = Some(16384);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert_eq!(p.effective.sched_chunk, PREFILL_CHUNK_FENCE);
        assert_eq!(p.effective.sched_chunk_live, PREFILL_CHUNK_FENCE);
        assert!(p.issues.iter().any(|i| i.code == "chunk_fenced"));
        assert!(!p.has_errors());
        assert!(p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_CONT_PREFILL_CHUNK" && v == "8192"));

        // The documented escape lifts the same cap for the plan.
        let mut req = ServingRequest::default();
        req.sched_chunk = Some(16384);
        req.chunk_fence = ChunkFence::Off;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert_eq!(p.effective.sched_chunk, 16384);
        assert!(!p.issues.iter().any(|i| i.code == "chunk_fenced"));
    }

    #[test]
    fn a_refused_dspark_drafter_is_an_error() {
        let facts = EngineFacts {
            dspark_ok: Some(false),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &ServingRequest::default(),
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "dspark_artifact"));
    }

    #[test]
    fn an_unloadable_artifact_is_an_error() {
        let facts = EngineFacts {
            artifact_ok: Some(false),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &ServingRequest::default(),
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "artifact_invalid"));
    }

    #[test]
    fn a_refused_vision_artifact_is_an_error() {
        let facts = EngineFacts {
            vision_path_ok: Some(false),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &ServingRequest::default(),
            Some(caps(ModelFamily::Inkling, Variant::InklingSmall)),
            &facts,
        );
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "vision_artifact"));
    }

    #[test]
    fn the_draft_length_is_reported_only_when_mtp_runs() {
        let mut req = ServingRequest::default();
        req.mtp_draft = Some(3);
        req.mtp_path = Some("mtp.gguf".into());
        let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
        assert_eq!(p.effective.mtp_draft, Some(3));
        assert_eq!(p.to_json()["effective"]["mtp_draft"], 3);

        let mut req = ServingRequest::default();
        req.mtp_draft = Some(3);
        req.mtp_mode = MtpMode::Off;
        let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
        assert_eq!(p.effective.mtp_draft, None);
        assert!(p.to_json()["effective"]["mtp_draft"].is_null());
    }

    #[test]
    fn the_two_persistence_thresholds_stay_apart() {
        let mut req = ServingRequest::default();
        req.kv_min_tokens = Some(512);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        // The disk store's record minimum is not the bank threshold.
        assert_eq!(p.effective.disk_min_tokens, Some(512));
        assert_eq!(p.effective.bank_persist_min, DEFAULT_BANK_PERSIST);
        assert_eq!(p.to_json()["effective"]["bank_persist_min_tokens"], 8192);
        assert_eq!(p.to_json()["effective"]["disk_min_tokens"], 512);

        let mut req = ServingRequest::default();
        req.bank_persist_min = Some(4096);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert_eq!(p.effective.bank_persist_min, 4096);
    }

    #[test]
    fn a_negative_persist_threshold_resolves_to_zero() {
        // `env_i32_bound` clamps it to zero at runtime, which disables
        // persistence; the plan must not report 8,192 instead.
        let mut req = ServingRequest::default();
        req.bank_persist_min = Some(0);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert_eq!(p.effective.bank_persist_min, 0);
        assert_eq!(p.to_json()["effective"]["bank_persist_min_tokens"], 0);
    }

    #[test]
    fn an_empty_sidecar_path_is_not_weights() {
        let mut req = ServingRequest::default();
        req.mtp_mode = MtpMode::On;
        req.mtp_path = Some(String::new());
        let facts = EngineFacts {
            mtp_path_ok: Some(false),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &req,
            Some(caps(ModelFamily::Step37, Variant::Step37Flash)),
            &facts,
        );
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "mtp_sidecar"));
        assert!(!p.effective.mtp_weights);
    }

    #[test]
    fn the_legacy_switch_keeps_banks_but_drops_the_driver() {
        // It forces the static/serial route, so the banks the static lane
        // coalesces over stay, while nothing enters the bank driver.
        let mut req = ServingRequest::default();
        req.lane = LaneMode::Serial;
        req.mtp_mode = MtpMode::On;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert_eq!(p.effective.max_seqs, 2);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "mtp_lane"));

        let mut req = ServingRequest::default();
        req.lane = LaneMode::Serial;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
        assert!(!p.has_errors());
    }

    #[test]
    fn the_legacy_switch_outranks_a_named_width() {
        // The switch disables continuous routing; the width only sizes the
        // banks the static lane coalesces over.
        let mut req = ServingRequest::default();
        req.lane = LaneMode::Serial;
        req.max_seqs = MaxSeqs::Fixed(2);
        req.mtp_mode = MtpMode::On;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p.wants_bank_lane());
        assert_eq!(p.effective.max_seqs, 2);
        assert!(p.issues.iter().any(|i| i.code == "mtp_lane"));
        assert!(!p
            .env_overrides()
            .iter()
            .any(|(k, _)| k == "DS4_SERVER_CONTINUOUS"));
    }

    #[test]
    fn auto_leaves_the_legacy_lane_switch_alone() {
        // README: DS4_SERVER_CONTINUOUS=0 forces the static/serial route.
        // Only a named width overrides it.
        let p = plan(
            ServingRequest::default(),
            ModelFamily::Qwen4Exp,
            Variant::Qwen38FlashNext,
        );
        assert_eq!(p.effective.max_seqs, 2);
        assert!(!p.wants_bank_lane());
        assert!(!p
            .env_overrides()
            .iter()
            .any(|(k, _)| k == "DS4_SERVER_CONTINUOUS"));
    }

    #[test]
    fn a_graph_family_cannot_serve_a_slice() {
        let mut req = ServingRequest::default();
        req.distribution = Distribution::Sliced;
        let p = plan(req, ModelFamily::SolarOpen2, Variant::SolarOpen2_250B);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "family_host"));
    }

    #[test]
    fn a_draft_below_the_family_minimum_cannot_speculate() {
        // Qwen allocates no speculative runtime at draft 1.
        let mut req = ServingRequest::default();
        req.mtp_mode = MtpMode::On;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p.has_errors());
        assert_eq!(p.effective.mtp_draft, Some(2));

        let mut req = ServingRequest::default();
        req.mtp_mode = MtpMode::On;
        req.mtp_draft = Some(1);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "mtp_draft"));
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);

        // Auto downgrades instead of failing.
        let mut req = ServingRequest::default();
        req.mtp_draft = Some(1);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p.has_errors());
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
        assert!(p.issues.iter().any(|i| i.code == "mtp_draft"));
    }

    #[test]
    fn a_graph_family_rejects_cpu_but_keeps_metal() {
        for family in [
            (ModelFamily::SolarOpen2, Variant::SolarOpen2_250B),
            (ModelFamily::ExaoneMoe, Variant::Kexaone236B),
            (ModelFamily::ExaoneMoe, Variant::K2Horizon375B),
        ] {
            let mut req = ServingRequest::default();
            req.backend = crate::Backend::Cpu;
            let p = plan(req, family.0, family.1);
            assert!(p.has_errors(), "{:?}", family.1);
            assert!(p.issues.iter().any(|i| i.code == "family_host"));

            let mut req = ServingRequest::default();
            req.backend = crate::Backend::Metal;
            let p = plan(req, family.0, family.1);
            assert!(!p.issues.iter().any(|i| i.code == "family_host"));
        }
    }

    #[test]
    fn the_qwen_yarn_cap_is_not_the_qualified_ctx() {
        let mut req = ServingRequest::default();
        req.ctx = 262_144 * QWEN_YARN_MAX_FACTOR as i32;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p.issues.iter().any(|i| i.code == "ctx_unavailable"));

        let mut req = ServingRequest::default();
        req.ctx = 262_144 * QWEN_YARN_MAX_FACTOR as i32 + 1;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "ctx_unavailable"));
    }

    #[test]
    fn ling_yarn_two_bank_plan() {
        for ctx in [131_072, 131_073, 262_144] {
            let req = ServingRequest {
                ctx,
                max_seqs: MaxSeqs::Fixed(2),
                backend: Backend::Cuda,
                ..ServingRequest::default()
            };
            let p = plan(req, ModelFamily::Ling3Vl, Variant::Ling30FlashVl);
            assert!(!p.has_errors(), "ctx={ctx}: {:?}", p.issues);
            assert_eq!(p.effective.max_seqs, 2);
            // A larger runtime cap does not extend measured qualification.
            assert_eq!(p.qualified.ctx, Some(65536));
        }

        let req = ServingRequest {
            ctx: 262_145,
            backend: Backend::Cuda,
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::Ling3Vl, Variant::Ling30FlashVl);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "ctx_unavailable"));
    }

    #[test]
    fn a_cuda_only_family_rejects_another_host() {
        for family in [
            (ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext),
            (ModelFamily::Inkling, Variant::InklingSmall),
            (ModelFamily::Step37, Variant::Step37Flash),
        ] {
            let mut req = ServingRequest::default();
            req.backend = crate::Backend::Cpu;
            let p = plan(req, family.0, family.1);
            assert!(p.has_errors());
            assert!(p.issues.iter().any(|i| i.code == "family_host"));

            let mut req = ServingRequest::default();
            req.distribution = Distribution::Sliced;
            let p = plan(req, family.0, family.1);
            assert!(p.issues.iter().any(|i| i.code == "family_host"));
        }
        // Families the open accepts elsewhere are untouched.
        let mut req = ServingRequest::default();
        req.distribution = Distribution::Sliced;
        let p = plan(req, ModelFamily::DeepSeek4, Variant::Flash);
        assert!(!p.issues.iter().any(|i| i.code == "family_host"));
    }

    #[test]
    fn a_ctx_above_the_session_cap_is_an_error() {
        // GLM's default 8,192 cannot create a session at all.
        let p = plan(
            ServingRequest::default(),
            ModelFamily::Glm53,
            Variant::Glm53Flash,
        );
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "ctx_unavailable"));

        let mut req = ServingRequest::default();
        req.ctx = 2048;
        let p = plan(req, ModelFamily::Glm53, Variant::Glm53Flash);
        assert!(!p.issues.iter().any(|i| i.code == "ctx_unavailable"));
    }

    #[test]
    fn a_nonpositive_ctx_is_an_error() {
        for ctx in [0, -1] {
            let mut req = ServingRequest::default();
            req.ctx = ctx;
            let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
            assert!(p.has_errors(), "ctx {ctx}");
            assert!(p.issues.iter().any(|i| i.code == "ctx_invalid"));
        }
    }

    #[test]
    fn glm_exact_context_boundary() {
        for ctx in [1, 2048, 2049, i32::MAX] {
            let req = ServingRequest {
                ctx,
                ..ServingRequest::default()
            };
            let p = plan(req, ModelFamily::Glm53, Variant::Glm53Flash);
            assert_eq!(p.has_errors(), ctx > 2048, "ctx={ctx}: {:?}", p.issues);
            assert_eq!(
                p.issues.iter().any(|i| i.code == "ctx_unavailable"),
                ctx > 2048
            );
            assert_eq!(p.qualified.ctx, Some(2048));
        }
    }

    #[test]
    fn native_capacity_is_not_yield() {
        let req = ServingRequest {
            sched_chunk: Some(512),
            native_chunk: Some(2048),
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
        assert_eq!(p.to_json()["effective"]["native_chunk"], 2048);
        assert_eq!(p.effective.sched_chunk, 512);
        let p = plan(
            ServingRequest::default(),
            ModelFamily::Step37,
            Variant::Step37Flash,
        );
        assert!(p.to_json()["effective"]["native_chunk"].is_null());
    }

    #[test]
    fn deepseek_batch_uses_native_cap() {
        for variant in [Variant::Flash, Variant::Pro] {
            let req = ServingRequest {
                native_chunk: Some(256),
                ctx: 8192,
                ..ServingRequest::default()
            };
            let p = plan(req, ModelFamily::DeepSeek4, variant);
            assert_eq!(p.batch_max_total_tokens(8192, 2), 256);
            assert!(p
                .env_overrides()
                .contains(&("DS4_METAL_PREFILL_CHUNK".into(), "256".into())));
        }
    }

    #[test]
    fn solar_batch_tokens_use_native_cap() {
        let mut req = ServingRequest::default();
        req.native_chunk = Some(256);
        req.ctx = 8192;
        let p = plan(req, ModelFamily::SolarOpen2, Variant::SolarOpen2_250B);
        assert!(
            p.env_overrides()
                .iter()
                .any(|(k, v)| k == "DS4_METAL_PREFILL_CHUNK" && v == "256"),
            "{:?}",
            p.env_overrides()
        );
        let width = p.effective.max_seqs as i32;
        let arg = p.batch_max_total_tokens(p.effective.ctx, width);
        assert_eq!(arg, 256);
        assert_ne!(arg, p.effective.ctx.saturating_mul(width));
    }

    #[test]
    fn qwen_auto_uses_qualified_partial() {
        let p = plan(
            ServingRequest::default(),
            ModelFamily::Qwen4Exp,
            Variant::Qwen38FlashNext,
        );
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Partial);
        assert_eq!(p.qualified.prefix_reuse, Support::Qualified);
        assert!(!p.has_errors());
    }

    #[test]
    fn exaone_partial_is_present() {
        let mut req = ServingRequest::default();
        req.prefix_reuse = PrefixReuse::Partial;
        let p = plan(req.clone(), ModelFamily::ExaoneMoe, Variant::Kexaone236B);
        assert!(!p.has_errors());
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Partial);
        assert_eq!(p.qualified.prefix_reuse, Support::Present);

        req.prefix_reuse = PrefixReuse::Auto;
        let p = plan(req, ModelFamily::ExaoneMoe, Variant::Kexaone236B);
        assert!(!p.has_errors());
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
        assert!(p.issues.iter().any(|i| i.code == "partial_unqualified"));
    }

    #[test]
    fn forced_partial_without_a_checkpoint_store_errors() {
        let mut req = ServingRequest::default();
        req.prefix_reuse = PrefixReuse::Partial;
        let facts = EngineFacts {
            partial_reuse: Some(false),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &req,
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "partial_runtime"));
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
    }

    #[test]
    fn a_downgraded_reuse_keeps_the_family_verification() {
        let facts = EngineFacts {
            partial_reuse: Some(false),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &ServingRequest::default(),
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
        assert_eq!(p.qualified.prefix_reuse, Support::Qualified);
    }

    #[test]
    fn auto_partial_without_a_checkpoint_store_warns() {
        let facts = EngineFacts {
            partial_reuse: Some(false),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &ServingRequest::default(),
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert!(!p.has_errors());
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
        assert!(p.issues.iter().any(|i| i.code == "partial_runtime"));
    }

    #[test]
    fn opt_in_banks_at_width_one_have_no_partial_lane() {
        // Step publishes DS4_STEP37_BATCH=0 there, so --check-config has to
        // predict the same refusal the refit would raise.
        for width in [MaxSeqs::Auto, MaxSeqs::Fixed(1)] {
            let mut req = ServingRequest::default();
            req.prefix_reuse = PrefixReuse::Partial;
            req.max_seqs = width;
            let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
            assert!(p.has_errors(), "{}", width.as_str());
            assert!(p.issues.iter().any(|i| i.code == "partial_lane"));
            assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
        }
        let mut req = ServingRequest::default();
        req.prefix_reuse = PrefixReuse::Partial;
        req.max_seqs = MaxSeqs::Fixed(2);
        let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
        assert!(!p.has_errors());
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Partial);
    }

    #[test]
    fn serial_alias_rejects_forced_partial() {
        let mut req = ServingRequest::default();
        req.prefix_reuse = PrefixReuse::Partial;
        req.max_seqs = MaxSeqs::Off;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "partial_lane"));
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
        assert!(p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_SERVER_FORK_PARTIAL" && v == "0"));
    }

    #[test]
    fn exaone_auto_stays_exact() {
        let p = plan(
            ServingRequest::default(),
            ModelFamily::ExaoneMoe,
            Variant::Kexaone236B,
        );
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
        assert!(!p.has_errors());
    }

    #[test]
    fn glm_disk_is_an_error() {
        let mut req = ServingRequest::default();
        req.ctx = 2048;
        req.kv_disk_dir = Some("/tmp/kv".into());
        let p = plan(req, ModelFamily::Glm53, Variant::Glm53Flash);
        assert!(p.has_errors());
        assert!(!p.effective.disk);
        assert!(p.issues.iter().any(|i| i.code == "disk_unsupported"));
        assert!(!p.issues.iter().any(|i| i.code == "ctx_unavailable"));
    }

    #[test]
    fn k2_mtp_on_is_an_error() {
        let mut req = ServingRequest::default();
        req.mtp_mode = MtpMode::On;
        let p = plan(req, ModelFamily::ExaoneMoe, Variant::K2Horizon375B);
        assert!(p.has_errors());
        assert!(!p.effective.mtp_weights);
    }

    #[test]
    fn k2_disk_is_warning_not_silent_success() {
        let mut req = ServingRequest::default();
        req.kv_disk_dir = Some("/tmp/kv".into());
        let p = plan(req, ModelFamily::ExaoneMoe, Variant::K2Horizon375B);
        assert!(!p.has_errors());
        assert!(p.effective.disk);
        assert_eq!(p.qualified.disk, Support::Present);
        assert!(p.issues.iter().any(|i| i.code == "disk_unverified"));
    }

    #[test]
    fn serial_default_auto_is_width_one() {
        // Each family's own session cap, since the shared default context
        // cannot create a GLM or Inkling session at all.
        let families = [
            (ModelFamily::Inkling, Variant::InklingSmall, 1024),
            (ModelFamily::Glm53, Variant::Glm53Flash, 2048),
            (ModelFamily::Dots3Note, Variant::Dots3NotePrev, DEFAULT_CTX),
        ];
        for (family, variant, ctx) in families {
            for req in [
                ServingRequest {
                    ctx,
                    ..ServingRequest::default()
                },
                ServingRequest {
                    ctx,
                    max_seqs: MaxSeqs::Auto,
                    ..ServingRequest::default()
                },
            ] {
                let p = plan(req, family, variant);
                assert!(
                    !p.has_errors(),
                    "{:?} default/auto must boot: {:?}",
                    family,
                    p.issues
                );
                assert!(
                    p.issues.iter().all(|i| i.code != "banks_unsupported"),
                    "{:?} default/auto must not be banks_unsupported",
                    family
                );
                assert_eq!(p.effective.max_seqs, 1);
            }
        }
    }

    #[test]
    fn step_default_auto_stays_serial() {
        for req in [
            ServingRequest::default(),
            ServingRequest {
                max_seqs: MaxSeqs::Auto,
                ..ServingRequest::default()
            },
        ] {
            let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
            assert!(!p.has_errors());
            assert_eq!(p.effective.max_seqs, 1);
            assert!(!p.effective.banks_opt_in);
            assert!(
                !p.env_overrides()
                    .iter()
                    .any(|(k, v)| k == "DS4_STEP37_BATCH" && v == "1"),
                "Step default/auto must not publish DS4_STEP37_BATCH=1"
            );
        }
    }

    #[test]
    fn step_max_seqs_enables_opt_in_banks() {
        let mut req = ServingRequest::default();
        req.max_seqs = MaxSeqs::Fixed(2);
        req.ctx = 65536;
        let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
        assert!(p.effective.banks_opt_in);
        assert_eq!(p.effective.max_seqs, 2);
        let env = p.env_overrides();
        assert!(env.iter().any(|(k, v)| k == "DS4_STEP37_BATCH" && v == "1"));
        assert!(env.iter().any(|(k, _)| k == "DS4_MEM_FLOOR_GB"));
        assert!(p.issues.iter().any(|i| i.code == "prompt_bound"));
    }

    #[test]
    fn qwen_max_seqs_publishes_batch_env() {
        let p = plan(
            ServingRequest::default(),
            ModelFamily::Qwen4Exp,
            Variant::Qwen38FlashNext,
        );
        assert!(p
            .env_overrides()
            .iter()
            .any(|(key, value)| key == "DS4_QWEN_BATCH" && value == "1"));

        let mut req = ServingRequest::default();
        req.max_seqs = MaxSeqs::Fixed(1);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p
            .env_overrides()
            .iter()
            .any(|(key, value)| key == "DS4_QWEN_BATCH" && value == "1"));
    }

    #[test]
    fn dots3_banks_are_present() {
        let req = ServingRequest {
            max_seqs: MaxSeqs::Fixed(2),
            prefix_reuse: PrefixReuse::Partial,
            mtp_mode: MtpMode::Off,
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::Dots3Note, Variant::Dots3NotePrev);
        assert!(!p.has_errors(), "{:?}", p.issues);
        assert_eq!(p.effective.max_seqs, 2);
        assert_eq!(p.qualified.banks, Support::Present);
        assert_eq!(p.qualified.prefix_reuse, Support::Present);
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Partial);
        assert!(p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_DOTS3_BATCH" && v == "1"));
    }

    #[test]
    fn dots3_mtp_is_serial_opt_in() {
        let req = ServingRequest {
            mtp_mode: MtpMode::On,
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::Dots3Note, Variant::Dots3NotePrev);
        assert!(!p.has_errors(), "{:?}", p.issues);
        assert_eq!(p.effective.mtp_mode, MtpMode::On);
        assert_eq!(p.qualified.mtp, Support::Present);
        assert!(p.issues.iter().any(|i| i.code == "mtp_unverified"));
        assert!(p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_DOTS3_MTP" && v == "1"));

        let p = plan(
            ServingRequest::default(),
            ModelFamily::Dots3Note,
            Variant::Dots3NotePrev,
        );
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
        assert!(p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_DOTS3_MTP" && v == "0"));
    }

    #[test]
    fn dots3_mtp_routes_serial() {
        for max_seqs in [MaxSeqs::Auto, MaxSeqs::Fixed(1)] {
            let req = ServingRequest {
                max_seqs,
                mtp_mode: MtpMode::On,
                ..ServingRequest::default()
            };
            let p = plan(req, ModelFamily::Dots3Note, Variant::Dots3NotePrev);
            assert!(!p.has_errors(), "{:?}", p.issues);
            assert_eq!(p.effective.mtp_mode, MtpMode::On);
            assert_eq!(p.effective.max_seqs, 1);
            assert!(p.uses_serial_mtp());
            assert!(!p.wants_bank_lane(), "{max_seqs:?}");
            let env = p.env_overrides();
            assert!(env.iter().any(|(k, v)| k == "DS4_DOTS3_BATCH" && v == "0"));
            assert!(
                env.iter()
                    .any(|(k, v)| k == "DS4_SERVER_CONTINUOUS" && v == "0"),
                "serial MTP must disable inherited continuous routing: {max_seqs:?}"
            );
        }
    }

    #[test]
    fn bank_mtp_keeps_routing() {
        for (family, variant, width, sidecar) in [
            (ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext, 1, None),
            (ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext, 2, None),
            (
                ModelFamily::Step37,
                Variant::Step37Flash,
                2,
                Some("step-mtp.gguf"),
            ),
        ] {
            let req = ServingRequest {
                max_seqs: MaxSeqs::Fixed(width),
                mtp_mode: MtpMode::On,
                mtp_path: sidecar.map(str::to_owned),
                ..ServingRequest::default()
            };
            let p = plan(req, family, variant);
            assert!(!p.has_errors(), "{:?}", p.issues);
            assert_eq!(p.effective.mtp_mode, MtpMode::On);
            assert!(!p.uses_serial_mtp());
            assert!(p.wants_bank_lane());
            assert!(p
                .env_overrides()
                .iter()
                .any(|(k, v)| k == "DS4_SERVER_CONTINUOUS" && v == "1"));
        }
    }

    #[test]
    fn dots3_mtp_refuses_bank_lane() {
        let req = ServingRequest {
            max_seqs: MaxSeqs::Fixed(2),
            mtp_mode: MtpMode::On,
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::Dots3Note, Variant::Dots3NotePrev);
        assert!(p.issues.iter().any(|i| i.code == "mtp_lane"));
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
        assert!(p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_DOTS3_MTP" && v == "0"));
    }

    #[test]
    fn dots3_mtp_refuses_long_draft() {
        let req = ServingRequest {
            mtp_mode: MtpMode::On,
            mtp_draft: Some(4),
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::Dots3Note, Variant::Dots3NotePrev);
        assert!(p.issues.iter().any(|i| i.code == "mtp_draft"));
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
    }

    #[test]
    fn dots3_partial_needs_bank_lane() {
        let mut req = ServingRequest::default();
        req.prefix_reuse = PrefixReuse::Partial;
        let p = plan(req, ModelFamily::Dots3Note, Variant::Dots3NotePrev);
        assert!(p.issues.iter().any(|i| i.code == "partial_lane"));
        let p = plan(
            ServingRequest::default(),
            ModelFamily::Dots3Note,
            Variant::Dots3NotePrev,
        );
        assert!(!p.has_errors(), "{:?}", p.issues);
        assert_eq!(p.effective.max_seqs, 1);
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
    }

    #[test]
    fn inkling_disk_is_present() {
        let mut req = ServingRequest::default();
        req.ctx = 1024;
        req.kv_disk_dir = Some("/tmp/kv".into());
        let p = plan(req, ModelFamily::Inkling, Variant::InklingSmall);
        assert!(!p.has_errors(), "{:?}", p.issues);
        assert!(p.effective.disk);
        assert_eq!(p.qualified.disk, Support::Present);
        assert!(p.issues.iter().any(|i| i.code == "disk_unverified"));
    }

    #[test]
    fn mem_floor_is_one_value() {
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 12;
        let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
        assert_eq!(p.requested.mem_floor_gb, 12);
        assert_eq!(p.effective.mem_floor_gb, 12);
        assert_eq!(
            p.env_overrides()
                .into_iter()
                .find(|(k, _)| k == "DS4_MEM_FLOOR_GB")
                .map(|(_, v)| v)
                .as_deref(),
            Some("12")
        );
    }

    #[test]
    fn parse_space_alias_and_max_seqs() {
        assert_eq!(parse_disk_space("32G").unwrap(), 32 * 1024);
        assert_eq!(parse_disk_space("32768").unwrap(), 32768);
        assert_eq!(parse_disk_space("8TiB").unwrap(), 8 * 1024 * 1024);
        // The alias must refuse what `--kv-disk-space-mb` refuses; the store
        // turns MiB into bytes.
        assert!(parse_disk_space("4096TiB").is_err());
        assert!(parse_disk_space("0").is_err());
        assert_eq!(MaxSeqs::parse("auto").unwrap(), MaxSeqs::Auto);
        assert_eq!(MaxSeqs::parse("2").unwrap(), MaxSeqs::Fixed(2));
        assert!(PrefixReuse::parse("maybe").is_err());
    }

    #[test]
    fn requested_effective_qualified_are_distinct() {
        let mut req = ServingRequest::default();
        req.prefix_reuse = PrefixReuse::Auto;
        req.mtp_mode = MtpMode::On;
        req.mtp_path = Some("mtp.gguf".into());
        req.ctx = 65536;
        req.max_seqs = MaxSeqs::Fixed(2);
        let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
        assert_eq!(p.requested.prefix_reuse, PrefixReuse::Auto);
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Partial);
        assert_eq!(p.qualified.prompt, Some(6300));
        assert_ne!(p.effective.ctx as u32, p.qualified.prompt.unwrap());
        assert!(p.effective.mtp_weights);
        let text = p.to_json().to_string();
        assert!(text.contains("\"requested\""));
        assert!(text.contains("\"effective\""));
        assert!(text.contains("\"qualified\""));
    }

    #[test]
    fn glm_forced_exact_reuse_errors() {
        let mut req = ServingRequest::default();
        req.prefix_reuse = PrefixReuse::Exact;
        let p = plan(req, ModelFamily::Glm53, Variant::Glm53Flash);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "reuse_unsupported"));
    }

    #[test]
    fn deepseek_mtp_on_without_path_errors() {
        let mut req = ServingRequest::default();
        req.mtp_mode = MtpMode::On;
        let p = plan(req, ModelFamily::DeepSeek4, Variant::Flash);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "mtp_sidecar"));
        assert!(!p.effective.mtp_weights);
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
    }

    #[test]
    fn max_seqs_zero_is_invalid() {
        assert!(MaxSeqs::parse("0").is_err());
        assert!(MaxSeqs::parse("65").is_err());
        assert_eq!(MaxSeqs::parse("1").unwrap(), MaxSeqs::Fixed(1));
        assert_eq!(MaxSeqs::parse_coalesce("0").unwrap(), MaxSeqs::Off);
    }

    #[test]
    fn coalesce_zero_keeps_serial() {
        let mut req = ServingRequest::default();
        req.max_seqs = MaxSeqs::Off;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p.has_errors());
        assert_eq!(p.effective.max_seqs, 1);
        assert!(!p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_QWEN_BATCH" && v == "1"));
        // Re-reading the published alias must not resurrect the lane.
        let published = p
            .env_overrides()
            .into_iter()
            .find(|(k, _)| k == "DS4_SERVER_COALESCE_MAX")
            .map(|(_, v)| v)
            .unwrap();
        assert_eq!(MaxSeqs::parse_coalesce(&published).unwrap(), MaxSeqs::Off);
    }

    #[test]
    fn serial_alias_rejects_forced_bank_mtp() {
        let mut req = ServingRequest::default();
        req.max_seqs = MaxSeqs::Off;
        req.mtp_mode = MtpMode::On;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "mtp_lane"));
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
        assert!(!p.effective.mtp_weights);
    }

    #[test]
    fn serial_alias_keeps_step_mtp() {
        let mut req = ServingRequest::default();
        req.max_seqs = MaxSeqs::Off;
        req.mtp_mode = MtpMode::On;
        req.mtp_path = Some("step-mtp.gguf".into());
        let p = plan(req, ModelFamily::Step37, Variant::Step37Flash);
        assert!(!p.has_errors());
        assert_eq!(p.effective.mtp_mode, MtpMode::On);
    }

    #[test]
    fn a_refused_lane_disables_bank_mtp() {
        let facts = EngineFacts {
            cont_lane: Some(false),
            banks_fitted: Some(1),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &ServingRequest::default(),
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert!(!p.has_errors());
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
        assert!(p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_MTP_SPEC_DISABLE" && v == "1"));
    }

    #[test]
    fn missing_mtp_path_is_an_error() {
        // The host opens the named artifact whatever the mode asks for.
        for mode in [MtpMode::On, MtpMode::Auto, MtpMode::Off] {
            let mut req = ServingRequest::default();
            req.mtp_mode = mode;
            req.mtp_path = Some("missing.gguf".into());
            let facts = EngineFacts {
                mtp_path_ok: Some(false),
                ..EngineFacts::default()
            };
            let p = resolve_plan(
                &req,
                Some(caps(ModelFamily::DeepSeek4, Variant::Flash)),
                &facts,
            );
            assert!(p.has_errors(), "{}", mode.as_str());
            assert!(p.issues.iter().any(|i| i.code == "mtp_sidecar"));
            assert!(!p.effective.mtp_weights);
        }
    }

    #[test]
    fn a_broken_mtp_path_outranks_the_backend_note() {
        let mut req = ServingRequest::default();
        req.backend = crate::Backend::Cpu;
        req.mtp_mode = MtpMode::Auto;
        req.mtp_path = Some("missing.gguf".into());
        let facts = EngineFacts {
            mtp_path_ok: Some(false),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &req,
            Some(caps(ModelFamily::Step37, Variant::Step37Flash)),
            &facts,
        );
        assert!(p.issues.iter().any(|i| i.code == "mtp_sidecar"));
    }

    #[test]
    fn an_explicit_width_wants_the_lane() {
        for width in [1, 2] {
            let mut req = ServingRequest::default();
            req.max_seqs = MaxSeqs::Fixed(width);
            let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
            assert!(p.wants_bank_lane(), "width {width}");
            assert!(p
                .env_overrides()
                .iter()
                .any(|(k, v)| k == "DS4_SERVER_CONTINUOUS" && v == "1"));
        }
        let mut req = ServingRequest::default();
        req.max_seqs = MaxSeqs::Off;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p.wants_bank_lane());
        // Auto on a serial family expresses no preference.
        let p = plan(
            ServingRequest::default(),
            ModelFamily::Step37,
            Variant::Step37Flash,
        );
        assert!(!p.wants_bank_lane());
        let mut req = ServingRequest::default();
        req.backend = crate::Backend::Cpu;
        req.max_seqs = MaxSeqs::Fixed(1);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p.wants_bank_lane());
    }

    #[test]
    fn cpu_backend_rejects_forced_cuda_banks() {
        let mut req = ServingRequest::default();
        req.backend = crate::Backend::Cpu;
        req.max_seqs = MaxSeqs::Fixed(2);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "banks_cuda"));
        assert_eq!(p.effective.max_seqs, 1);
    }

    #[test]
    fn cpu_backend_auto_stays_serial() {
        // DeepSeek is the one family the native CPU session accepts.
        let mut req = ServingRequest::default();
        req.backend = crate::Backend::Cpu;
        let p = plan(req, ModelFamily::DeepSeek4, Variant::Flash);
        assert!(!p.has_errors());
        assert_eq!(p.effective.max_seqs, 1);
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
        assert!(!p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_SERVER_CONTINUOUS" && v == "1"));
    }

    #[test]
    fn metal_mtp_on_errors() {
        let mut req = ServingRequest::default();
        req.backend = crate::Backend::Metal;
        req.mtp_mode = MtpMode::On;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "mtp_cuda"));
        assert!(!p.effective.mtp_weights);
    }

    #[test]
    fn disk_open_failed_is_an_error() {
        let mut req = ServingRequest::default();
        req.kv_disk_dir = Some("/tmp/kv".into());
        let facts = EngineFacts {
            disk_ready: Some(false),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &req,
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "disk_open"));
        assert!(!p.effective.disk);
    }

    #[test]
    fn a_chunk_outside_the_set_is_not_yield() {
        let mut req = ServingRequest::default();
        req.sched_chunk = Some(3000);
        req.sched_chunk_live = Some(700);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert_eq!(p.effective.sched_chunk, 2048);
        assert_eq!(p.effective.sched_chunk_live, 512);
        assert!(VERIFIED_PREFILL_CHUNKS.contains(&p.effective.sched_chunk));
        assert!(VERIFIED_PREFILL_CHUNKS.contains(&p.effective.sched_chunk_live));
        assert_ne!(p.effective.sched_chunk, 3000);
        assert!(!p.has_errors());
    }

    #[test]
    fn published_chunk_never_exceeds_native() {
        let mut req = ServingRequest::default();
        req.sched_chunk = Some(8192);
        req.sched_chunk_live = Some(8192);
        req.native_chunk = Some(1024);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert_eq!(p.effective.native_chunk, Some(1024));
        assert_eq!(p.effective.sched_chunk, 1024);
        assert_eq!(p.effective.sched_chunk_live, 1024);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "chunk_past_native"));
        assert!(!p.may_listen());
        let env = p.env_overrides();
        assert!(env
            .iter()
            .any(|(k, v)| k == "DS4_CONT_PREFILL_CHUNK" && v == "1024"));
        assert!(env
            .iter()
            .any(|(k, v)| k == "DS4_CONT_PREFILL_CHUNK_LIVE" && v == "1024"));
        assert!(p.effective.sched_chunk <= 1024);
        assert!(p.effective.sched_chunk_live <= 1024);
    }

    #[test]
    fn explicit_live_past_native_errors() {
        // Boot yield sits at native; only the live flag is the oversize yield.
        let mut req = ServingRequest::default();
        req.sched_chunk = Some(1024);
        req.sched_chunk_live = Some(8192);
        req.native_chunk = Some(1024);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "chunk_past_native"));
        assert!(!p.may_listen());
        assert!(p.effective.sched_chunk_live <= 1024);
        let env = p.env_overrides();
        assert!(env.iter().any(|(k, v)| {
            k == "DS4_CONT_PREFILL_CHUNK_LIVE" && v.parse::<u32>().unwrap() <= 1024
        }));
        assert!(!env.iter().any(|(k, v)| {
            k == "DS4_CONT_PREFILL_CHUNK_LIVE" && v.parse::<u32>().unwrap() > 1024
        }));
    }

    #[test]
    fn short_native_tail_is_valid() {
        for cap in [64, 128, 255] {
            let req = ServingRequest {
                ctx: cap as i32,
                native_chunk: Some(cap),
                ..ServingRequest::default()
            };
            let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
            assert!(p.may_listen(), "{:?}", p.issues);
            assert_eq!(p.effective.sched_chunk, cap);
            assert_eq!(p.effective.sched_chunk_live, cap);
        }
    }

    #[test]
    fn a_chunk_below_the_set_is_an_error() {
        let mut req = ServingRequest::default();
        req.sched_chunk = Some(128);
        req.native_chunk = Some(1024);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "chunk_unverified"));
        assert!(!p.may_listen());
        assert_ne!(p.effective.sched_chunk, 128);
        assert_eq!(p.effective.sched_chunk, VERIFIED_PREFILL_CHUNKS[0]);
        let env = p.env_overrides();
        assert!(!env
            .iter()
            .any(|(k, v)| k == "DS4_CONT_PREFILL_CHUNK" && v == "128"));
        assert!(env.iter().any(|(k, v)| {
            k == "DS4_CONT_PREFILL_CHUNK"
                && v.parse::<u32>().ok() == Some(VERIFIED_PREFILL_CHUNKS[0])
        }));
    }

    #[test]
    fn a_live_chunk_below_the_set_is_an_error() {
        let mut req = ServingRequest::default();
        req.sched_chunk = Some(1024);
        req.sched_chunk_live = Some(128);
        req.native_chunk = Some(1024);
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "chunk_unverified"));
        assert!(!p.may_listen());
        assert_ne!(p.effective.sched_chunk_live, 128);
        assert_eq!(p.effective.sched_chunk_live, VERIFIED_PREFILL_CHUNKS[0]);
        let env = p.env_overrides();
        assert!(!env
            .iter()
            .any(|(k, v)| k == "DS4_CONT_PREFILL_CHUNK_LIVE" && v == "128"));
        assert!(env.iter().any(|(k, v)| {
            k == "DS4_CONT_PREFILL_CHUNK_LIVE"
                && v.parse::<u32>().ok() == Some(VERIFIED_PREFILL_CHUNKS[0])
        }));
    }

    #[test]
    fn auto_quote_covers_each_budget() {
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 4;
        let facts = EngineFacts {
            host_available_bytes: Some(100 * GIB),
            shared_weights_bytes: Some(10 * GIB),
            per_bank_bytes: Some(8 * GIB),
            mtp_state_bytes: Some(2 * GIB),
            scratch_bytes: Some(1 * GIB),
            checkpoint_pool_bytes: Some(1 * GIB),
            ple_bytes: Some(1 * GIB),
            media_reserve_bytes: Some(4 * GIB),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &req,
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        let q = p.quote.expect("quote");
        assert_eq!(q.shared_weights, 10 * GIB);
        assert_eq!(q.per_bank, 8 * GIB);
        assert_eq!(q.mtp_state, 2 * GIB);
        assert_eq!(q.scratch, 1 * GIB);
        assert_eq!(q.checkpoint_pool, 1 * GIB);
        assert_eq!(q.ple, 1 * GIB);
        assert_eq!(q.media_reserve, 4 * GIB);
        assert_eq!(q.floor, 4 * GIB);
        assert_eq!(q.available, 100 * GIB);
        assert_eq!(p.effective.max_seqs, 2);
        assert_eq!(q.banks, 2);
        assert_eq!(q.total, q.cost(2));
        assert!(q.total <= q.available);
        let json = p.to_json();
        assert_eq!(json["quote"]["shared_weights"], 10 * GIB);
        assert_eq!(json["quote"]["per_bank"], 8 * GIB);
        assert_eq!(json["quote"]["mtp_state"], 2 * GIB);
        assert_eq!(json["quote"]["scratch"], 1 * GIB);
        assert_eq!(json["quote"]["checkpoint_pool"], 1 * GIB);
        assert_eq!(json["quote"]["ple"], 1 * GIB);
        assert_eq!(json["quote"]["media_reserve"], 4 * GIB);
        assert_eq!(json["quote"]["floor"], 4 * GIB);
        assert!(!p.has_errors());
    }

    #[test]
    fn media_serial_auto_is_not_two_text_banks() {
        let mut caps = caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        caps.media_serial = true;
        let p = resolve_plan(
            &ServingRequest::default(),
            Some(caps),
            &EngineFacts::default(),
        );
        assert_eq!(p.effective.max_seqs, 1);
        assert!(!p.has_errors());
    }

    #[test]
    fn auto_quote_keeps_the_media_reserve() {
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 0;
        let facts = EngineFacts {
            host_available_bytes: Some(40 * GIB),
            shared_weights_bytes: Some(10 * GIB),
            per_bank_bytes: Some(10 * GIB),
            media_reserve_bytes: Some(20 * GIB),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &req,
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert_eq!(p.effective.max_seqs, 1);
        let q = p.quote.expect("quote");
        assert_eq!(q.media_reserve, 20 * GIB);
        assert_eq!(q.banks, 1);
        assert!(!p.has_errors());
    }

    #[test]
    fn auto_quote_that_cannot_host_is_an_error() {
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 4;
        let facts = EngineFacts {
            host_available_bytes: Some(10 * GIB),
            shared_weights_bytes: Some(8 * GIB),
            per_bank_bytes: Some(8 * GIB),
            ..EngineFacts::default()
        };
        let p = resolve_plan(
            &req,
            Some(caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext)),
            &facts,
        );
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "quote_overflow"));
        assert!(!p.may_listen());
        assert_eq!(p.effective.max_seqs, 1);
    }
}
