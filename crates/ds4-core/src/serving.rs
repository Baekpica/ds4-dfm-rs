//! Common serving contract: requested / effective / qualified.
//!
//! Names are shared. Implementation cost and verification stay per family.
//! Forced options that a family cannot run become errors, not silent fallback.

use crate::identify::Identified;
use crate::shape::{ModelFamily, Shape, Variant};
use crate::Backend;
use serde_json::{json, Value};
use std::fmt::{self, Write as _};

pub const DEFAULT_MEM_FLOOR_GB: u64 = 4;
pub const DEFAULT_MAX_SEQS: u32 = 2;
pub const DEFAULT_CTX: i32 = 8192;
pub const DEFAULT_SCHED_CHUNK: u32 = 4096;
pub const DEFAULT_SCHED_LIVE: u32 = 512;
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
    pub kv_min_tokens: Option<i32>,
    pub mtp_path: Option<String>,
    pub mtp_draft: Option<i32>,
    pub sched_chunk: Option<u32>,
    pub sched_chunk_live: Option<u32>,
    pub native_chunk: Option<u32>,
    pub print_plan: bool,
    pub check_config: bool,
    pub backend: Backend,
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
    /// `Some(false)` once the native fit refused the continuous lane.
    pub cont_lane: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestedView {
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
    pub max_seqs: u32,
    pub ctx: i32,
    pub mem_floor_gb: u64,
    pub disk: bool,
    pub banks_opt_in: bool,
    pub sched_chunk: u32,
    pub sched_chunk_live: u32,
    pub bank_persist_min: i32,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedPlan {
    pub family: Option<ModelFamily>,
    pub variant: Option<Variant>,
    pub family_name: &'static str,
    pub caps: Option<ServingCaps>,
    pub requested: RequestedView,
    pub effective: EffectiveView,
    pub qualified: QualifiedView,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestTrace {
    pub effective_lane: &'static str,
    pub reuse_kind: ReuseTaken,
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
            mtp_path: None,
            mtp_draft: None,
            sched_chunk: None,
            sched_chunk_live: None,
            native_chunk: None,
            print_plan: false,
            check_config: false,
            backend: Backend::Cuda,
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
    /// Compatible aliases: `DS4_SERVER_COALESCE_MAX`, `DS4_MEM_FLOOR_GB`,
    /// `DS4_SERVER_FORK`, `DS4_SERVER_FORK_PARTIAL`, chunk env vars.
    pub fn from_env() -> Self {
        let mut req = Self::default();
        if let Ok(raw) = std::env::var("DS4_SERVER_COALESCE_MAX") {
            if let Ok(parsed) = MaxSeqs::parse_coalesce(&raw) {
                req.max_seqs = parsed;
            }
        }
        if let Ok(raw) = std::env::var("DS4_MEM_FLOOR_GB") {
            if let Some(gb) = parse_u64_atoi(&raw) {
                req.mem_floor_gb = gb;
            }
        }
        req.prefix_reuse = reuse_from_env();
        if std::env::var_os("DS4_MTP_SPEC_DISABLE").is_some() {
            req.mtp_mode = MtpMode::Off;
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
    match unit.as_str() {
        "" | "m" | "mb" | "mi" | "mib" => Ok(n),
        "g" | "gb" | "gi" | "gib" => Ok(n.saturating_mul(1024)),
        "t" | "tb" | "ti" | "tib" => Ok(n.saturating_mul(1024 * 1024)),
        "k" | "kb" | "ki" | "kib" => Ok(n.saturating_add(1023) / 1024),
        _ => Err(format!("ds4-server-rs: invalid disk space unit in '{raw}'")),
    }
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
            qualified_ctx: Some(32768),
            qualified_banks: Some(1),
            qualified_prompt: None,
            media_serial: false,
        };
    }
    match family {
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
            spec_lane: SpecLane::Serial,
            qualified_ctx: Some(65536),
            qualified_banks: Some(2),
            qualified_prompt: Some(6300),
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
            reuse: ReuseKind::Exact,
            reuse_support: Support::Qualified,
            disk: Support::Qualified,
            snapshot: Support::Qualified,
            mtp: MtpKind::None,
            mtp_support: Support::None,
            spec_lane: SpecLane::None,
            qualified_ctx: None,
            qualified_banks: None,
            qualified_prompt: None,
            media_serial: false,
        },
        ModelFamily::Dots3Note => ServingCaps {
            family,
            variant,
            banks: BankLane::Serial,
            bank_support: Support::None,
            reuse: ReuseKind::Exact,
            reuse_support: Support::Present,
            disk: Support::Qualified,
            snapshot: Support::Qualified,
            mtp: MtpKind::BoundOnly,
            mtp_support: Support::None,
            spec_lane: SpecLane::None,
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
            disk: Support::None,
            snapshot: Support::None,
            mtp: MtpKind::Sidecar,
            mtp_support: Support::Qualified,
            spec_lane: SpecLane::Serial,
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
        };
    };

    let reuse = resolve_reuse(req.prefix_reuse, caps, &mut issues);
    let (max_seqs, banks_opt_in) =
        resolve_seqs(req.max_seqs, caps, facts, req.backend, &mut issues);
    let (mtp_mode, mtp_weights) = resolve_mtp(req, caps, facts, &mut issues);
    let disk = resolve_disk(req, caps, facts, &mut issues);

    if req.ctx > 0 {
        if let Some(qctx) = caps.qualified_ctx {
            if req.ctx as u32 > qctx {
                issues.push(warn(
                    "ctx_unqualified",
                    format!("configured ctx {} exceeds qualified ctx {qctx}", req.ctx),
                ));
            }
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

    let sched_chunk = req.sched_chunk.unwrap_or(DEFAULT_SCHED_CHUNK);
    let mut sched_live = req.sched_chunk_live.unwrap_or(DEFAULT_SCHED_LIVE);
    if sched_live > sched_chunk {
        sched_live = sched_chunk;
    }

    let qualified = QualifiedView {
        prefix_reuse: if reuse == ReuseKind::None {
            Support::None
        } else if reuse == caps.reuse {
            caps.reuse_support
        } else {
            Support::Present
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
        ctx: caps.qualified_ctx,
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
            max_seqs,
            ctx: req.ctx,
            mem_floor_gb: req.mem_floor_gb,
            disk,
            banks_opt_in,
            sched_chunk,
            sched_chunk_live: sched_live,
            bank_persist_min: req.kv_min_tokens.unwrap_or(DEFAULT_BANK_PERSIST),
            native_chunk: req.native_chunk,
        },
        qualified,
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
        }
    }
}

impl ResolvedPlan {
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.level == IssueLevel::Error)
    }

    pub fn env_overrides(&self) -> Vec<(String, String)> {
        let mut out = vec![
            (
                "DS4_MEM_FLOOR_GB".into(),
                self.effective.mem_floor_gb.to_string(),
            ),
            (
                "DS4_SERVER_COALESCE_MAX".into(),
                self.effective.max_seqs.to_string(),
            ),
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
        if self.effective.max_seqs > 1 {
            out.push(("DS4_SERVER_CONTINUOUS".into(), "1".into()));
        }
        // 0/1 so a later fitted-down plan can retract. Step width 1 is
        // serial MTP; Qwen width 1 is still a bank.
        if self.family == Some(ModelFamily::Step37) {
            out.push((
                "DS4_STEP37_BATCH".into(),
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
        out.push((
            "DS4_CONT_PREFILL_CHUNK".into(),
            self.effective.sched_chunk.to_string(),
        ));
        out.push((
            "DS4_CONT_PREFILL_CHUNK_LIVE".into(),
            self.effective.sched_chunk_live.to_string(),
        ));
        out
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
            "requested": {
                "prefix_reuse": self.requested.prefix_reuse.as_str(),
                "mtp_mode": self.requested.mtp_mode.as_str(),
                "max_seqs": self.requested.max_seqs.as_str(),
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
                "max_seqs": self.effective.max_seqs,
                "ctx": self.effective.ctx,
                "mem_floor_gb": self.effective.mem_floor_gb,
                "disk": self.effective.disk,
                "banks_opt_in": self.effective.banks_opt_in,
                "sched_chunk": self.effective.sched_chunk,
                "sched_chunk_live": self.effective.sched_chunk_live,
                "native_chunk": self.effective.native_chunk,
                "bank_persist_min_tokens": self.effective.bank_persist_min,
                "disk_is_offload": false
            },
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
        json!({
            "effective_lane": self.effective_lane,
            "reuse_kind": self.reuse_kind.as_str(),
            "speculation_active": self.speculation_active,
            "fallback_reason": self.fallback_reason
        })
    }
}

fn default_effective(req: &ServingRequest) -> EffectiveView {
    EffectiveView {
        prefix_reuse: ReuseKind::None,
        mtp_mode: req.mtp_mode,
        mtp_weights: req.mtp_path.is_some(),
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
        bank_persist_min: req.kv_min_tokens.unwrap_or(DEFAULT_BANK_PERSIST),
        native_chunk: req.native_chunk,
    }
}

fn resolve_reuse(
    requested: PrefixReuse,
    caps: ServingCaps,
    issues: &mut Vec<PlanIssue>,
) -> ReuseKind {
    match requested {
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
                caps.reuse
            } else {
                ReuseKind::Partial
            }
        }
        PrefixReuse::Auto => caps.reuse,
    }
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
    let want = match requested {
        MaxSeqs::Off => 1,
        MaxSeqs::Auto => match caps.banks {
            BankLane::Serial | BankLane::OptIn => 1,
            BankLane::Persistent => caps.qualified_banks.unwrap_or(DEFAULT_MAX_SEQS),
        },
        MaxSeqs::Fixed(n) => n,
    };
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
    let opt_in = n > 1 && caps.banks == BankLane::OptIn;
    (n, opt_in)
}

fn resolve_mtp(
    req: &ServingRequest,
    caps: ServingCaps,
    facts: &EngineFacts,
    issues: &mut Vec<PlanIssue>,
) -> (MtpMode, bool) {
    let has_path = req.mtp_path.is_some();
    let can = match caps.mtp {
        MtpKind::None | MtpKind::BoundOnly => false,
        MtpKind::Embedded | MtpKind::Sidecar | MtpKind::DeepSeek => true,
    };
    if req.backend != Backend::Cuda {
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
    // Qwen/DeepSeek speculation lives in the bank driver. The legacy zero
    // alias (or a refused fit) routes every request through NativeDecode,
    // which only speculates for Inkling and Step, so MTP would never run.
    if caps.spec_lane == SpecLane::Bank && bank_lane_off(req, facts) {
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
    if has_path && facts.mtp_path_ok == Some(false) {
        if req.mtp_mode == MtpMode::On {
            issues.push(error(
                "mtp_sidecar",
                format!(
                    "{} MTP path is missing or is not a GGUF",
                    caps.variant_name()
                ),
            ));
        }
        return (MtpMode::Off, false);
    }
    if has_path && caps.mtp == MtpKind::None {
        issues.push(error(
            "mtp_contract",
            format!(
                "{} has no MTP contract; do not pass --mtp",
                caps.variant_name()
            ),
        ));
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
        MtpKind::Embedded => req.mtp_mode != MtpMode::Off,
        MtpKind::Sidecar | MtpKind::DeepSeek => has_path || facts.mtp_loaded,
        MtpKind::BoundOnly | MtpKind::None => false,
    };
    let mode = match req.mtp_mode {
        MtpMode::Off => MtpMode::Off,
        MtpMode::On if weights => MtpMode::On,
        MtpMode::On => MtpMode::Off,
        MtpMode::Auto if weights => MtpMode::Auto,
        MtpMode::Auto => MtpMode::Off,
    };
    (mode, weights)
}

/// The bank driver is absent when the operator forced serial through the
/// legacy zero alias or the native fit refused the lane.
fn bank_lane_off(req: &ServingRequest, facts: &EngineFacts) -> bool {
    req.max_seqs == MaxSeqs::Off || facts.cont_lane == Some(false)
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

fn qualified_note(caps: ServingCaps) -> &'static str {
    match caps.variant {
        Variant::Step37Flash => {
            "text banks are opt-in; Chat restart hits need history-stable identity; images serial"
        }
        Variant::K2Horizon375B => {
            "32K one-bank serving is qualified; disk KV and external owner import are not"
        }
        Variant::Glm53Flash => "serial graph is capped at 2,048 tokens; snapshots unsupported",
        Variant::Dots3NotePrev => "live serving is serial; embedded MTP is bound, not executed",
        Variant::InklingSmall => "serial exact-prefix reuse; disk snapshot unimplemented",
        Variant::Kexaone236B => "exact-frontier reuse only; partial checkpoint is a separate task",
        Variant::Qwen38FlashNext => {
            "common UX baseline; configured values and verified combinations differ"
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
    fn mtp_on_does_not_publish_disable() {
        let req = ServingRequest {
            mtp_mode: MtpMode::On,
            mtp_path: Some("mtp.gguf".into()),
            ..ServingRequest::default()
        };
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p
            .env_overrides()
            .iter()
            .any(|(key, _)| key == "DS4_MTP_SPEC_DISABLE"));
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
    fn exaone_forced_partial_is_an_error() {
        let mut req = ServingRequest::default();
        req.prefix_reuse = PrefixReuse::Partial;
        let p = plan(req, ModelFamily::ExaoneMoe, Variant::Kexaone236B);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "partial_unsupported"));
        assert_eq!(p.effective.prefix_reuse, ReuseKind::Exact);
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
        req.kv_disk_dir = Some("/tmp/kv".into());
        let p = plan(req, ModelFamily::Glm53, Variant::Glm53Flash);
        assert!(p.has_errors());
        assert!(!p.effective.disk);
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
        let families = [
            (ModelFamily::Inkling, Variant::InklingSmall),
            (ModelFamily::Glm53, Variant::Glm53Flash),
            (ModelFamily::Dots3Note, Variant::Dots3NotePrev),
        ];
        for (family, variant) in families {
            for req in [
                ServingRequest::default(),
                ServingRequest {
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
    fn dots3_banks_and_mtp_on_fail_honestly() {
        let mut req = ServingRequest::default();
        req.max_seqs = MaxSeqs::Fixed(2);
        req.mtp_mode = MtpMode::On;
        let p = plan(req, ModelFamily::Dots3Note, Variant::Dots3NotePrev);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "banks_unsupported"));
        assert!(p.issues.iter().any(|i| i.code == "mtp_unexecuted"));
    }

    #[test]
    fn inkling_disk_is_unsupported() {
        let mut req = ServingRequest::default();
        req.kv_disk_dir = Some("/tmp/kv".into());
        let p = plan(req, ModelFamily::Inkling, Variant::InklingSmall);
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "disk_unsupported"));
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
        let mut req = ServingRequest::default();
        req.mtp_mode = MtpMode::On;
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
        assert!(p.has_errors());
        assert!(p.issues.iter().any(|i| i.code == "mtp_sidecar"));
        assert!(!p.effective.mtp_weights);
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
        let mut req = ServingRequest::default();
        req.backend = crate::Backend::Cpu;
        let p = plan(req, ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        assert!(!p.has_errors());
        assert_eq!(p.effective.max_seqs, 1);
        assert_eq!(p.effective.mtp_mode, MtpMode::Off);
        assert!(!p
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_QWEN_BATCH" && v == "1"));
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
}
