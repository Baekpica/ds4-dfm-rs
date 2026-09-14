//! Host-owned `/metrics` / `/v1/stats` porcelain, including the memgov
//! census format from `ds4_server.c` at v0.6.3-dfm.
//!
//! Live CUDA/census numbers stay native; this crate owns the wire text.
//! `census_supported = false` follows C: the cell family is absent, not zero.

use crate::admit::{AdmitState, SHED_NAMES, SHED_REASONS};
use crate::route::{
    ThinkMode, WireSurface, LANE_CONTINUOUS, LANE_SERIAL, LANE_STATIC, REASON_NAMES,
};

pub const ROUTE_SURFACES: usize = 4;
pub const ROUTE_LANES: usize = 3;
pub const ROUTE_REASONS: usize = 14;
pub const THINK_MODES: usize = 4;

pub const ROUTE_SURFACE_NAMES: [&str; ROUTE_SURFACES] = [
    "openai_chat",
    "openai_completion",
    "anthropic_messages",
    "openai_responses",
];
pub const ROUTE_LANE_NAMES: [&str; ROUTE_LANES] = ["serial", "continuous", "static"];
pub const THINK_MODE_NAMES: [&str; THINK_MODES] = ["none", "low", "high", "max"];

#[derive(Debug, Clone, Default)]
pub struct RouteMetrics {
    pub route_requests: [[u64; ROUTE_LANES]; ROUTE_SURFACES],
    pub route_decisions: [u64; ROUTE_REASONS],
    pub shed: [u64; SHED_REASONS],
    pub think: [u64; THINK_MODES],
}

pub fn surface_index(surf: WireSurface) -> usize {
    surf as usize
}

pub fn lane_index(lane: u8) -> usize {
    match lane {
        LANE_SERIAL => 0,
        LANE_CONTINUOUS => 1,
        LANE_STATIC => 2,
        _ => 0,
    }
}

impl RouteMetrics {
    pub fn record_route(&mut self, surf: WireSurface, lane: u8, reason: u8, think: ThinkMode) {
        let si = surface_index(surf);
        let li = lane_index(lane);
        self.route_requests[si][li] += 1;
        if (reason as usize) < ROUTE_REASONS {
            self.route_decisions[reason as usize] += 1;
        }
        let ti = think as usize;
        if ti < THINK_MODES {
            self.think[ti] += 1;
        }
    }

    /// C `route_metrics_record`: one lane ENTRY tick alone (no decision or
    /// think tick). Lane-entry counters fire where a lane takes the job and
    /// can fire more than once per request — C's `cont_admit` success ticks
    /// continuous before the engine's funding verdict, and the stranded
    /// fallback's `generate_job` entry then ticks serial as well.
    pub fn record_lane_entry(&mut self, surf: WireSurface, lane: u8) {
        self.route_requests[surface_index(surf)][lane_index(lane)] += 1;
    }

    pub fn record_shed(&mut self, reason: u8) {
        if (reason as usize) < SHED_REASONS {
            self.shed[reason as usize] += 1;
        }
    }
}

/// Host-owned `/metrics` fragment (route / shed / think / queue gauges).
pub fn render_metrics_fragment(m: &RouteMetrics, admit: &AdmitState) -> String {
    let mut b = String::new();
    b.push_str("# TYPE ds4_route_requests_total counter\n");
    for si in 0..ROUTE_SURFACES {
        for li in 0..ROUTE_LANES {
            b.push_str(&format!(
                "ds4_route_requests_total{{surface=\"{}\",lane=\"{}\"}} {}\n",
                ROUTE_SURFACE_NAMES[si], ROUTE_LANE_NAMES[li], m.route_requests[si][li]
            ));
        }
    }
    b.push_str("# TYPE ds4_route_decisions_total counter\n");
    for ri in 0..ROUTE_REASONS {
        b.push_str(&format!(
            "ds4_route_decisions_total{{reason=\"{}\"}} {}\n",
            REASON_NAMES[ri], m.route_decisions[ri]
        ));
    }
    b.push_str("# TYPE ds4_requests_shed_total counter\n");
    for ri in 0..SHED_REASONS {
        b.push_str(&format!(
            "ds4_requests_shed_total{{reason=\"{}\"}} {}\n",
            SHED_NAMES[ri], m.shed[ri]
        ));
    }
    b.push_str("# TYPE ds4_requests_think_total counter\n");
    for ti in 0..THINK_MODES {
        b.push_str(&format!(
            "ds4_requests_think_total{{mode=\"{}\"}} {}\n",
            THINK_MODE_NAMES[ti], m.think[ti]
        ));
    }
    b.push_str(&format!(
        "# TYPE ds4_clients_connected gauge\nds4_clients_connected {}\n\
         # TYPE ds4_queue_depth gauge\nds4_queue_depth {}\n\
         # TYPE ds4_inflight_body_bytes gauge\nds4_inflight_body_bytes {}\n",
        admit.clients, admit.queued, admit.inflight_body_bytes
    ));
    b
}

/// Host-owned `/v1/stats` JSON: route observation plus C memory/governor siblings.
pub fn render_stats_json(m: &RouteMetrics, admit: &AdmitState) -> String {
    render_stats_json_ex(m, admit, &RuntimeMetrics::default())
}

pub fn render_stats_json_ex(m: &RouteMetrics, admit: &AdmitState, rt: &RuntimeMetrics) -> String {
    render_stats_json_plan(m, admit, rt, None, None)
}

/// GET /v1/stats body; plan and last_request are omitted when absent.
pub fn render_stats_json_plan(
    m: &RouteMetrics,
    admit: &AdmitState,
    rt: &RuntimeMetrics,
    serving: Option<&str>,
    last_request: Option<&str>,
) -> String {
    let mut b = render_stats_observation(m, admit);
    debug_assert!(b.ends_with("}\n"));
    b.truncate(b.len() - 2);
    b.push(',');
    b.push_str(&render_stats_memgov_json(&rt.memgov));
    if let Some(serving) = serving {
        b.push_str(",\"serving\":");
        b.push_str(serving);
    }
    if let Some(last_request) = last_request {
        b.push_str(",\"last_request\":");
        b.push_str(last_request);
    }
    b.push_str("}\n");
    b
}

fn render_stats_observation(m: &RouteMetrics, admit: &AdmitState) -> String {
    let mut b = String::from("{\"routes\":{");
    let mut first = true;
    for si in 0..ROUTE_SURFACES {
        for li in 0..ROUTE_LANES {
            if !first {
                b.push(',');
            }
            first = false;
            b.push_str(&format!(
                "\"{}_{}\":{}",
                ROUTE_SURFACE_NAMES[si], ROUTE_LANE_NAMES[li], m.route_requests[si][li]
            ));
        }
    }
    b.push_str("},\"route_decisions\":{");
    for ri in 0..ROUTE_REASONS {
        if ri > 0 {
            b.push(',');
        }
        b.push_str(&format!(
            "\"{}\":{}",
            REASON_NAMES[ri], m.route_decisions[ri]
        ));
    }
    b.push_str("},\"think_modes\":{");
    for ti in 0..THINK_MODES {
        if ti > 0 {
            b.push(',');
        }
        b.push_str(&format!("\"{}\":{}", THINK_MODE_NAMES[ti], m.think[ti]));
    }
    b.push_str("},\"sheds\":{");
    for ri in 0..SHED_REASONS {
        if ri > 0 {
            b.push(',');
        }
        b.push_str(&format!("\"{}\":{}", SHED_NAMES[ri], m.shed[ri]));
    }
    b.push_str(&format!(
        "}},\"clients\":{},\"queue_depth\":{},\"inflight_body_bytes\":{}}}\n",
        admit.clients, admit.queued, admit.inflight_body_bytes
    ));
    b
}

pub const MEM_CLASS_NAMES: [&str; 17] = [
    "engine_other",
    "weight_arena",
    "weight_span",
    "weight_whole",
    "weight_derived",
    "weight_import",
    "weight_artifact",
    "weight_host_pin",
    "stage_pin",
    "scalars_mirror",
    "session_tensors",
    "kv_primary",
    "batch_bank",
    "scratch_sticky",
    "kernel_partials",
    "graph_exec",
    "diag",
];
pub const MEM_DOMAIN_NAMES: [&str; 2] = ["unified_device", "pinned_host"];
pub const MEM_OBS_STATUS_NAMES: [&str; 3] = ["ok", "unsupported", "query_error"];
pub const MEM_OBS_SOURCE_NAMES: [&str; 3] = ["none", "cuda_free", "meminfo_available"];
pub const GOV_CONSUMER_NAMES: [&str; 5] = [
    "engine_boot",
    "prewarm",
    "batch_bank_plan",
    "serial_session",
    "static_batch",
];
pub const GOV_STATUS_NAMES: [&str; 6] = [
    "admit",
    "refuse_class",
    "refuse_live",
    "retry_obs",
    "unsupported",
    "fault",
];
pub const GOV_CMP_NAMES: [&str; 6] = [
    "agree",
    "live_stricter",
    "shadow_stricter",
    "verdict_class",
    "obs_policy",
    "fault",
];
pub const REJLANE_NAMES: [&str; 3] = ["continuous", "serial", "static"];
pub const REJECT_REASON_NAMES: [&str; 7] = [
    "class_budget",
    "live_headroom",
    "obs_retry",
    "unsupported",
    "fault",
    "deep_policy",
    "lane_disabled",
];
pub const RECLAIM_STATUS_NAMES: [&str; 6] = [
    "ok",
    "partial",
    "stale_plan",
    "busy",
    "unsupported",
    "device_error",
];
pub const RESUNIT_STATE_NAMES: [&str; 8] = [
    "populated",
    "already_ready",
    "satisfied_import",
    "host_mapped_by_policy",
    "lazy_deferred",
    "expert_cold_by_policy",
    "waived_optional",
    "failed",
];
pub const RESSTAGE_NAMES: [&str; 2] = ["boot", "lazy"];
pub const RESIDENCY_POLICY_NAMES: [&str; 4] = ["eager", "lazy", "mapped", "unattributed"];
pub const MODEL_ROLE_NAMES: [&str; 3] = ["primary", "auxiliary", "drafter"];

#[derive(Debug, Clone, Copy, Default)]
pub struct MemCell {
    pub requested: u64,
    pub committed: u64,
    pub freed_requested: u64,
    pub freed_committed: u64,
}

impl MemCell {
    pub fn live_requested(self) -> u64 {
        self.requested.wrapping_sub(self.freed_requested)
    }
    pub fn live(self) -> u64 {
        self.committed.wrapping_sub(self.freed_committed)
    }
    pub fn slack(self) -> u64 {
        let com = self.live();
        let req = self.live_requested();
        if com > req {
            com - req
        } else {
            0
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GovLease {
    pub intent: u64,
    pub resident: u64,
    pub reservation: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ReconcileSnap {
    pub residual: i64,
    pub onetime: u64,
}

#[derive(Debug, Clone)]
pub struct MemgovSnap {
    pub census_supported: bool,
    pub cells: [[MemCell; 2]; 17],
    pub census_faults: u64,
    pub torn_fallbacks: u64,
    pub census_epoch: u64,
    pub substrate_outstanding: u64,
    pub emit_substrate: bool,
    pub obs_status: u8,
    pub obs_source: u8,
    pub obs_free: u64,
    pub obs_total: u64,
    pub obs_cuda_free: u64,
    pub obs_meminfo: u64,
    pub memobs_calls: [u64; 3],
    pub memobs_errors: u64,
    pub own_trim_calls: u64,
    pub own_trim_recovered: u64,
    pub reconcile: Option<ReconcileSnap>,
    pub reconcile_flagged: u64,
    pub gov_epoch: u64,
    pub leases: [GovLease; 5],
    pub decisions: [[[u64; 6]; 6]; 5],
    pub deficit: [u64; 5],
    pub faults: u64,
    pub gov_modes: [&'static str; 5],
    pub residency_supported: bool,
    pub residency_units: [[u64; 8]; 4],
    pub residency_failures: [[u64; 2]; 3],
}

impl Default for MemgovSnap {
    fn default() -> Self {
        Self {
            census_supported: false,
            cells: [[MemCell::default(); 2]; 17],
            census_faults: 0,
            torn_fallbacks: 0,
            census_epoch: 0,
            substrate_outstanding: 0,
            emit_substrate: true,
            obs_status: 1,
            obs_source: 0,
            obs_free: 0,
            obs_total: 0,
            obs_cuda_free: 0,
            obs_meminfo: 0,
            memobs_calls: [0; 3],
            memobs_errors: 0,
            own_trim_calls: 0,
            own_trim_recovered: 0,
            reconcile: None,
            reconcile_flagged: 0,
            gov_epoch: 0,
            leases: [GovLease::default(); 5],
            decisions: [[[0; 6]; 6]; 5],
            deficit: [0; 5],
            faults: 0,
            gov_modes: ["enforce"; 5],
            residency_supported: false,
            residency_units: [[0; 8]; 4],
            residency_failures: [[0; 2]; 3],
        }
    }
}

pub fn parse_gov_mode(s: &str) -> Option<&'static str> {
    match s {
        "off" => Some("off"),
        "observe" => Some("observe"),
        "enforce" => Some("enforce"),
        _ => None,
    }
}

/// C `ds4_gov_modes_init` name board (defaults enforce; `DS4_MEMGOV` then per-family knobs).
pub fn gov_modes_from_env() -> [&'static str; 5] {
    const KNOBS: [&str; 5] = [
        "DS4_MEMGOV_BOOT",
        "DS4_MEMGOV_PREWARM",
        "DS4_MEMGOV_BANK",
        "DS4_MEMGOV_SERIAL",
        "DS4_MEMGOV_STATIC",
    ];
    let gdef = std::env::var("DS4_MEMGOV")
        .ok()
        .filter(|v| !v.is_empty())
        .and_then(|v| parse_gov_mode(&v));
    let mut out = ["enforce"; 5];
    for (i, knob) in KNOBS.iter().enumerate() {
        let mut m = gdef.unwrap_or("enforce");
        if let Ok(v) = std::env::var(knob) {
            if !v.is_empty() {
                if let Some(pm) = parse_gov_mode(&v) {
                    m = pm;
                }
            }
        }
        out[i] = m;
    }
    out
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeMetrics {
    pub uptime_seconds: u64,
    pub requests_started: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub requests_canceled: u64,
    pub requests_refused_deep_serial: u64,
    pub requests_inflight: u64,
    pub requests_serial: u64,
    pub out_backlog_bytes: u64,
    pub cont_admit_rejects: u64,
    pub cont_credit_ext_granted: u64,
    pub cont_credit_ext_refused: u64,
    pub have_batch_ctx: bool,
    pub cont_commit_observed: u64,
    pub cont_commit_packed: u64,
    pub cont_admit_band_x1024: u32,
    pub serial_idle_reaps: u64,
    pub cont_batch_failures: u64,
    pub graph_fit_refusals: u64,
    pub rejected: [[u64; 7]; 3],
    pub reclaim_banks: [u64; 6],
    pub reclaim_bytes: [u64; 6],
    pub admits_cold: u64,
    pub admits_warm: u64,
    pub admits_fork: u64,
    pub admits_partial_fork: u64,
    pub admits_partial_truncate: u64,
    pub tokens_prefilled_computed: u64,
    pub tokens_prefilled_cached: u64,
    pub tokens_decoded: u64,
    pub decode_steps: u64,
    pub spec_drafts: u64,
    pub spec_hits: u64,
    pub spec_quench: u64,
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    pub tok_per_step: f64,
    pub banks_live: u64,
    pub banks_total: u64,
    pub warm_records: u64,
    pub kv_pages_resident: u64,
    pub cont_ontoken_ns: u64,
    pub cont_ontoken_tokens: u64,
    pub batch_genmu_wait_ns: u64,
    pub batch_genmu_waits: u64,
    pub creg_published: u64,
    pub creg_resolved: u64,
    pub creg_missed: u64,
    pub creg_demoted: u64,
    pub creg_records_live: u64,
    pub artifact_source: u64,
    pub derived_artifacts: u64,
    pub derived_artifact_bytes: u64,
    pub memgov: MemgovSnap,
}

pub fn artifact_source_name(source: u64) -> &'static str {
    match source {
        3 => "mixed",
        2 => "built",
        1 => "imported",
        _ => "none",
    }
}

fn spec_accept_ratio(drafts: u64, hits: u64) -> f64 {
    if drafts == 0 {
        0.0
    } else {
        hits as f64 / drafts as f64
    }
}

/// C `/metrics` lines before the route/shed/think fragment.
pub fn render_metrics_prefix(rt: &RuntimeMetrics) -> String {
    format!(
        "# TYPE ds4_uptime_seconds gauge\n\
         ds4_uptime_seconds {}\n\
         # TYPE ds4_requests_started_total counter\n\
         ds4_requests_started_total {}\n\
         # TYPE ds4_requests_total counter\n\
         ds4_requests_total{{outcome=\"completed\"}} {}\n\
         ds4_requests_total{{outcome=\"failed\"}} {}\n\
         ds4_requests_total{{outcome=\"canceled\"}} {}\n\
         ds4_requests_total{{outcome=\"refused_deep_serial\"}} {}\n\
         # TYPE ds4_requests_inflight gauge\n\
         ds4_requests_inflight {}\n\
         # TYPE ds4_requests_serial_total counter\n\
         ds4_requests_serial_total {}\n",
        rt.uptime_seconds,
        rt.requests_started,
        rt.requests_completed,
        rt.requests_failed,
        rt.requests_canceled,
        rt.requests_refused_deep_serial,
        rt.requests_inflight,
        rt.requests_serial
    )
}

/// C `/metrics` lines after the queue gauges, before memgov census.
pub fn render_metrics_runtime(rt: &RuntimeMetrics) -> String {
    let mut b = String::new();
    b.push_str(&format!(
        "# TYPE ds4_out_backlog_bytes gauge\n\
         ds4_out_backlog_bytes {}\n\
         # TYPE ds4_cont_admit_rejects_total counter\n\
         ds4_cont_admit_rejects_total {}\n\
         # TYPE ds4_cont_credit_extension_granted_total counter\n\
         ds4_cont_credit_extension_granted_total {}\n\
         # TYPE ds4_cont_credit_extension_refused_total counter\n\
         ds4_cont_credit_extension_refused_total {}\n",
        rt.out_backlog_bytes,
        rt.cont_admit_rejects,
        rt.cont_credit_ext_granted,
        rt.cont_credit_ext_refused
    ));
    if rt.have_batch_ctx {
        b.push_str(&format!(
            "# TYPE ds4_cont_commit_bytes_per_token gauge\n\
             ds4_cont_commit_bytes_per_token{{kind=\"observed\"}} {}\n\
             ds4_cont_commit_bytes_per_token{{kind=\"packed\"}} {}\n\
             # TYPE ds4_cont_admit_band_x1024 gauge\n\
             ds4_cont_admit_band_x1024 {}\n",
            rt.cont_commit_observed, rt.cont_commit_packed, rt.cont_admit_band_x1024
        ));
    }
    b.push_str(&format!(
        "# TYPE ds4_serial_idle_reaps_total counter\n\
         ds4_serial_idle_reaps_total {}\n\
         # TYPE ds4_cont_batch_failures_total counter\n\
         ds4_cont_batch_failures_total {}\n\
         # TYPE ds4_graph_fit_refusals_total counter\n\
         ds4_graph_fit_refusals_total {}\n",
        rt.serial_idle_reaps, rt.cont_batch_failures, rt.graph_fit_refusals
    ));
    b.push_str("# TYPE ds4_requests_rejected_total counter\n");
    for (ln, lane) in REJLANE_NAMES.iter().enumerate() {
        for (rr, reason) in REJECT_REASON_NAMES.iter().enumerate() {
            b.push_str(&format!(
                "ds4_requests_rejected_total{{lane=\"{lane}\",reason=\"{reason}\"}} {}\n",
                rt.rejected[ln][rr]
            ));
        }
    }
    b.push_str("# TYPE ds4_reclaim_banks_total counter\n");
    for (rs, name) in RECLAIM_STATUS_NAMES.iter().enumerate() {
        b.push_str(&format!(
            "ds4_reclaim_banks_total{{result=\"{name}\"}} {}\n",
            rt.reclaim_banks[rs]
        ));
    }
    b.push_str("# TYPE ds4_reclaim_bytes_total counter\n");
    for (rs, name) in RECLAIM_STATUS_NAMES.iter().enumerate() {
        b.push_str(&format!(
            "ds4_reclaim_bytes_total{{result=\"{name}\"}} {}\n",
            rt.reclaim_bytes[rs]
        ));
    }
    let ratio = spec_accept_ratio(rt.spec_drafts, rt.spec_hits);
    b.push_str(&format!(
        "# TYPE ds4_admits_total counter\n\
         ds4_admits_total{{kind=\"cold\"}} {}\n\
         ds4_admits_total{{kind=\"warm\"}} {}\n\
         ds4_admits_total{{kind=\"fork\"}} {}\n\
         ds4_admits_total{{kind=\"partial_fork\"}} {}\n\
         ds4_admits_total{{kind=\"partial_truncate\"}} {}\n\
         # TYPE ds4_tokens_prefilled_total counter\n\
         ds4_tokens_prefilled_total{{kind=\"computed\"}} {}\n\
         ds4_tokens_prefilled_total{{kind=\"cached\"}} {}\n\
         # TYPE ds4_tokens_decoded_total counter\n\
         ds4_tokens_decoded_total {}\n\
         # TYPE ds4_decode_steps_total counter\n\
         ds4_decode_steps_total {}\n\
         # TYPE ds4_spec_drafts_total counter\n\
         ds4_spec_drafts_total {}\n\
         # TYPE ds4_spec_hits_total counter\n\
         ds4_spec_hits_total {}\n\
         # TYPE ds4_spec_quench_total counter\n\
         ds4_spec_quench_total {}\n\
         # TYPE ds4_spec_accept_ratio gauge\n\
         ds4_spec_accept_ratio {:.4}\n\
         # TYPE ds4_decode_tok_s gauge\n\
         ds4_decode_tok_s {:.2}\n\
         # TYPE ds4_prefill_tok_s gauge\n\
         ds4_prefill_tok_s {:.2}\n\
         # TYPE ds4_tok_per_step gauge\n\
         ds4_tok_per_step {:.3}\n\
         # TYPE ds4_banks_live gauge\n\
         ds4_banks_live {}\n\
         # TYPE ds4_banks_total gauge\n\
         ds4_banks_total {}\n\
         # TYPE ds4_warm_records gauge\n\
         ds4_warm_records {}\n\
         # TYPE ds4_kv_pages_resident gauge\n\
         ds4_kv_pages_resident {}\n\
         # TYPE ds4_cont_ontoken_ns_total counter\n\
         ds4_cont_ontoken_ns_total {}\n\
         # TYPE ds4_cont_ontoken_tokens_total counter\n\
         ds4_cont_ontoken_tokens_total {}\n\
         # TYPE ds4_batch_genmu_wait_ns_total counter\n\
         ds4_batch_genmu_wait_ns_total {}\n\
         # TYPE ds4_batch_genmu_waits_total counter\n\
         ds4_batch_genmu_waits_total {}\n\
         # TYPE ds4_continuation_records_published_total counter\n\
         ds4_continuation_records_published_total {}\n\
         # TYPE ds4_continuation_resolved_total counter\n\
         ds4_continuation_resolved_total {}\n\
         # TYPE ds4_continuation_missed_total counter\n\
         ds4_continuation_missed_total {}\n\
         # TYPE ds4_continuation_demoted_total counter\n\
         ds4_continuation_demoted_total {}\n\
         # TYPE ds4_continuation_records_live gauge\n\
         ds4_continuation_records_live {}\n\
         # TYPE ds4_derived_artifacts gauge\n\
         ds4_derived_artifacts{{source=\"{}\"}} {}\n\
         # TYPE ds4_derived_artifact_bytes gauge\n\
         ds4_derived_artifact_bytes {}\n",
        rt.admits_cold,
        rt.admits_warm,
        rt.admits_fork,
        rt.admits_partial_fork,
        rt.admits_partial_truncate,
        rt.tokens_prefilled_computed,
        rt.tokens_prefilled_cached,
        rt.tokens_decoded,
        rt.decode_steps,
        rt.spec_drafts,
        rt.spec_hits,
        rt.spec_quench,
        ratio,
        rt.decode_tok_s,
        rt.prefill_tok_s,
        rt.tok_per_step,
        rt.banks_live,
        rt.banks_total,
        rt.warm_records,
        rt.kv_pages_resident,
        rt.cont_ontoken_ns,
        rt.cont_ontoken_tokens,
        rt.batch_genmu_wait_ns,
        rt.batch_genmu_waits,
        rt.creg_published,
        rt.creg_resolved,
        rt.creg_missed,
        rt.creg_demoted,
        rt.creg_records_live,
        artifact_source_name(rt.artifact_source),
        rt.derived_artifacts,
        rt.derived_artifact_bytes
    ));
    b
}

/// C memgov D0a/D0b/D5 `/metrics` family.
pub fn render_memgov_metrics(g: &MemgovSnap) -> String {
    let mut b = String::new();
    b.push_str(&format!(
        "# TYPE ds4_memory_census_supported gauge\n\
         ds4_memory_census_supported {}\n",
        u32::from(g.census_supported)
    ));
    if g.census_supported {
        b.push_str("# TYPE ds4_memory_bytes gauge\n");
        for (d, domain) in MEM_DOMAIN_NAMES.iter().enumerate() {
            for (c, class) in MEM_CLASS_NAMES.iter().enumerate() {
                let cell = g.cells[c][d];
                b.push_str(&format!(
                    "ds4_memory_bytes{{domain=\"{domain}\",class=\"{class}\",state=\"requested\"}} {}\n\
                     ds4_memory_bytes{{domain=\"{domain}\",class=\"{class}\",state=\"allocated\"}} {}\n\
                     ds4_memory_bytes{{domain=\"{domain}\",class=\"{class}\",state=\"slack\"}} {}\n",
                    cell.live_requested(),
                    cell.live(),
                    cell.slack()
                ));
            }
        }
        b.push_str(&format!(
            "# TYPE ds4_memory_census_faults_total counter\n\
             ds4_memory_census_faults_total {}\n",
            g.census_faults
        ));
    }
    b.push_str(&format!(
        "# TYPE ds4_memory_census_torn_fallbacks_total counter\n\
         ds4_memory_census_torn_fallbacks_total {}\n\
         # TYPE ds4_memory_census_epoch gauge\n\
         ds4_memory_census_epoch {}\n",
        g.torn_fallbacks, g.census_epoch
    ));
    if g.emit_substrate {
        b.push_str(&format!(
            "# TYPE ds4_memory_substrate_outstanding_bytes gauge\n\
             ds4_memory_substrate_outstanding_bytes {}\n",
            g.substrate_outstanding
        ));
    }
    b.push_str("# TYPE ds4_memory_observation_info gauge\n");
    for (st, status) in MEM_OBS_STATUS_NAMES.iter().enumerate() {
        for (so, source) in MEM_OBS_SOURCE_NAMES.iter().enumerate() {
            let one = i32::from(g.obs_status as usize == st && g.obs_source as usize == so);
            b.push_str(&format!(
                "ds4_memory_observation_info{{status=\"{status}\",source=\"{source}\"}} {one}\n"
            ));
        }
    }
    if g.obs_status == 0 {
        b.push_str(&format!(
            "# TYPE ds4_memory_observation_bytes gauge\n\
             ds4_memory_observation_bytes{{kind=\"free\"}} {}\n\
             ds4_memory_observation_bytes{{kind=\"total\"}} {}\n\
             ds4_memory_observation_bytes{{kind=\"cuda_free\"}} {}\n\
             ds4_memory_observation_bytes{{kind=\"meminfo_available\"}} {}\n",
            g.obs_free, g.obs_total, g.obs_cuda_free, g.obs_meminfo
        ));
    }
    b.push_str("# TYPE ds4_memory_observation_calls_total counter\n");
    for (so, source) in MEM_OBS_SOURCE_NAMES.iter().enumerate() {
        b.push_str(&format!(
            "ds4_memory_observation_calls_total{{source=\"{source}\"}} {}\n",
            g.memobs_calls[so]
        ));
    }
    b.push_str(&format!(
        "# TYPE ds4_memory_observation_errors_total counter\n\
         ds4_memory_observation_errors_total {}\n\
         # TYPE ds4_mem_own_trim_calls_total counter\n\
         ds4_mem_own_trim_calls_total {}\n\
         # TYPE ds4_mem_own_trim_recovered_bytes_total counter\n\
         ds4_mem_own_trim_recovered_bytes_total {}\n",
        g.memobs_errors, g.own_trim_calls, g.own_trim_recovered
    ));
    if let Some(rr) = g.reconcile {
        b.push_str(&format!(
            "# TYPE ds4_mem_reconcile_residual_bytes gauge\n\
             ds4_mem_reconcile_residual_bytes {}\n\
             # TYPE ds4_mem_reconcile_onetime_bytes gauge\n\
             ds4_mem_reconcile_onetime_bytes {}\n",
            rr.residual, rr.onetime
        ));
    }
    b.push_str(&format!(
        "# TYPE ds4_mem_reconcile_flagged_total counter\n\
         ds4_mem_reconcile_flagged_total {}\n\
         # TYPE ds4_memory_governor_epoch gauge\n\
         ds4_memory_governor_epoch {}\n",
        g.reconcile_flagged, g.gov_epoch
    ));
    b.push_str("# TYPE ds4_memory_lease_bytes gauge\n");
    for (c, consumer) in GOV_CONSUMER_NAMES.iter().enumerate() {
        let l = g.leases[c];
        b.push_str(&format!(
            "ds4_memory_lease_bytes{{consumer=\"{consumer}\",field=\"intent\"}} {}\n\
             ds4_memory_lease_bytes{{consumer=\"{consumer}\",field=\"resident\"}} {}\n\
             ds4_memory_lease_bytes{{consumer=\"{consumer}\",field=\"reservation\"}} {}\n",
            l.intent, l.resident, l.reservation
        ));
    }
    b.push_str("# TYPE ds4_memory_decisions_total counter\n");
    for (c, consumer) in GOV_CONSUMER_NAMES.iter().enumerate() {
        for (st, result) in GOV_STATUS_NAMES.iter().enumerate() {
            for (r, reason) in GOV_CMP_NAMES.iter().enumerate() {
                b.push_str(&format!(
                    "ds4_memory_decisions_total{{consumer=\"{consumer}\",result=\"{result}\",reason=\"{reason}\"}} {}\n",
                    g.decisions[c][st][r]
                ));
            }
        }
    }
    b.push_str("# TYPE ds4_memory_decision_deficit_bytes gauge\n");
    for (c, consumer) in GOV_CONSUMER_NAMES.iter().enumerate() {
        b.push_str(&format!(
            "ds4_memory_decision_deficit_bytes{{consumer=\"{consumer}\"}} {}\n",
            g.deficit[c]
        ));
    }
    b.push_str(&format!(
        "# TYPE ds4_memory_governor_faults_total counter\n\
         ds4_memory_governor_faults_total {}\n",
        g.faults
    ));
    if g.residency_supported {
        b.push_str("# TYPE ds4_residency_units counter\n");
        for (p, policy) in RESIDENCY_POLICY_NAMES.iter().enumerate() {
            for (st, state) in RESUNIT_STATE_NAMES.iter().enumerate() {
                b.push_str(&format!(
                    "ds4_residency_units{{policy=\"{policy}\",state=\"{state}\"}} {}\n",
                    g.residency_units[p][st]
                ));
            }
        }
        b.push_str("# TYPE ds4_residency_failures_total counter\n");
        for (r, role) in MODEL_ROLE_NAMES.iter().enumerate() {
            for (sg, stage) in RESSTAGE_NAMES.iter().enumerate() {
                b.push_str(&format!(
                    "ds4_residency_failures_total{{model_role=\"{role}\",stage=\"{stage}\"}} {}\n",
                    g.residency_failures[r][sg]
                ));
            }
        }
    }
    b
}

/// Full C `/metrics` body (prefix + route fragment + runtime + memgov).
pub fn render_metrics(m: &RouteMetrics, admit: &AdmitState, rt: &RuntimeMetrics) -> String {
    let mut b = render_metrics_prefix(rt);
    b.push_str(&render_metrics_fragment(m, admit));
    b.push_str(&render_metrics_runtime(rt));
    b.push_str(&render_memgov_metrics(&rt.memgov));
    b
}

pub fn render_stats_memgov_json(g: &MemgovSnap) -> String {
    let mut b = format!(
        "\"memory\":{{\"census_supported\":{},\"census_epoch\":{}",
        if g.census_supported { "true" } else { "false" },
        g.census_epoch
    );
    if g.census_supported {
        b.push_str(",\"classes\":{");
        let mut tl = [0u64; 2];
        let mut ts = [0u64; 2];
        for (c, class) in MEM_CLASS_NAMES.iter().enumerate() {
            if c > 0 {
                b.push(',');
            }
            b.push_str(&format!("\"{class}\":{{"));
            for (d, domain) in MEM_DOMAIN_NAMES.iter().enumerate() {
                if d > 0 {
                    b.push(',');
                }
                let lv = g.cells[c][d].live();
                let sl = g.cells[c][d].slack();
                tl[d] += lv;
                ts[d] += sl;
                b.push_str(&format!("\"{domain}\":{{\"live\":{lv},\"slack\":{sl}}}"));
            }
            b.push('}');
        }
        b.push_str(&format!(
            "}},\"totals\":{{\"device_live\":{},\"device_slack\":{},\"host_pin_live\":{},\"host_pin_slack\":{}}},\"census_faults\":{}",
            tl[0], ts[0], tl[1], ts[1], g.census_faults
        ));
        if g.emit_substrate {
            b.push_str(&format!(
                ",\"substrate_outstanding\":{}",
                g.substrate_outstanding
            ));
        }
    }
    let st = MEM_OBS_STATUS_NAMES[g.obs_status.min(2) as usize];
    let so = MEM_OBS_SOURCE_NAMES[g.obs_source.min(2) as usize];
    b.push_str(&format!(
        ",\"observation\":{{\"status\":\"{st}\",\"source\":\"{so}\""
    ));
    if g.obs_status == 0 {
        b.push_str(&format!(
            ",\"free_bytes\":{},\"total_bytes\":{},\"cuda_free_bytes\":{},\"meminfo_available_bytes\":{}",
            g.obs_free, g.obs_total, g.obs_cuda_free, g.obs_meminfo
        ));
    }
    b.push_str(&format!(
        ",\"calls_cuda_free\":{},\"calls_meminfo_available\":{},\"errors\":{}}}",
        g.memobs_calls[1], g.memobs_calls[2], g.memobs_errors
    ));
    b.push_str(",\"reconcile\":{\"supported\":false}}");
    b.push_str(&format!(
        ",\"governor\":{{\"shadow\":true,\"epoch\":{},\"faults\":{},\"modes\":{{",
        g.gov_epoch, g.faults
    ));
    for (c, consumer) in GOV_CONSUMER_NAMES.iter().enumerate() {
        if c > 0 {
            b.push(',');
        }
        b.push_str(&format!("\"{consumer}\":\"{}\"", g.gov_modes[c]));
    }
    b.push_str("},\"leases\":{");
    for (c, consumer) in GOV_CONSUMER_NAMES.iter().enumerate() {
        if c > 0 {
            b.push(',');
        }
        let l = g.leases[c];
        b.push_str(&format!(
            "\"{consumer}\":{{\"intent\":{},\"resident\":{},\"reservation\":{}}}",
            l.intent, l.resident, l.reservation
        ));
    }
    b.push_str("},\"decisions\":{");
    for (r, reason) in GOV_CMP_NAMES.iter().enumerate() {
        if r > 0 {
            b.push(',');
        }
        let mut n = 0u64;
        for c in 0..5 {
            for st in 0..6 {
                n += g.decisions[c][st][r];
            }
        }
        b.push_str(&format!("\"{reason}\":{n}"));
    }
    b.push_str("}}");
    b
}

pub fn dump_memgov_names() -> String {
    let mut b = String::new();
    for n in MEM_CLASS_NAMES {
        b.push_str(&format!("CLASS {n}\n"));
    }
    for n in MEM_DOMAIN_NAMES {
        b.push_str(&format!("DOMAIN {n}\n"));
    }
    for n in GOV_CONSUMER_NAMES {
        b.push_str(&format!("GOV {n}\n"));
    }
    for n in GOV_STATUS_NAMES {
        b.push_str(&format!("STATUS {n}\n"));
    }
    for n in GOV_CMP_NAMES {
        b.push_str(&format!("CMP {n}\n"));
    }
    for n in REJLANE_NAMES {
        b.push_str(&format!("REJLANE {n}\n"));
    }
    for n in REJECT_REASON_NAMES {
        b.push_str(&format!("REJECT {n}\n"));
    }
    for n in RECLAIM_STATUS_NAMES {
        b.push_str(&format!("RECLAIM {n}\n"));
    }
    for n in MEM_OBS_STATUS_NAMES {
        b.push_str(&format!("OBS_STATUS {n}\n"));
    }
    for n in MEM_OBS_SOURCE_NAMES {
        b.push_str(&format!("OBS_SOURCE {n}\n"));
    }
    for n in RESIDENCY_POLICY_NAMES {
        b.push_str(&format!("POLICY {n}\n"));
    }
    for n in RESUNIT_STATE_NAMES {
        b.push_str(&format!("RESUNIT {n}\n"));
    }
    for n in RESSTAGE_NAMES {
        b.push_str(&format!("RESSTAGE {n}\n"));
    }
    for n in MODEL_ROLE_NAMES {
        b.push_str(&format!("ROLE {n}\n"));
    }
    b
}
