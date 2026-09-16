//! Host-owned session ledger: token timeline, generation, sync plan.
//!
//! Copied from `ds4_session_sync` / `rewrite_*` / `rewind` / `invalidate`
//! at v0.6.3-dfm. Native CUDA still executes prefill/decode; this module
//! decides reuse vs rebuild and is the authoritative pos/generation/ctx.

use crate::shape::{ModelFamily, Shape};

pub const PREFILL_FENCE_ROWS: u32 = 8192;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionBackend {
    Cuda,
    Cpu,
}

impl SessionBackend {
    pub fn from_oracle_name(s: &str) -> Option<Self> {
        match s {
            "cuda" => Some(Self::Cuda),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    pub fn oracle_name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewriteKind {
    Error,
    Extend,
    Rebuild,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncPlan {
    pub err: bool,
    pub start: i32,
    pub rebuild: bool,
    pub bump: bool,
    pub fence: bool,
    pub bounds: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RewindResult {
    pub pos: i32,
    pub bump: bool,
    pub solar_invalid: bool,
    pub valid: bool,
    pub generation: u64,
}

#[derive(Clone, Debug)]
pub struct SessionLedger {
    pub family: ModelFamily,
    pub backend: SessionBackend,
    pub ctx: i32,
    pub prefill_cap: u32,
    tokens: Vec<i32>,
    pub valid: bool,
    pub generation: u64,
    pub solar_state_valid: bool,
    pub mtp_draft_valid: bool,
    pub n_swa: u32,
    pub n_swa_period: u32,
    pub n_layer: u32,
    pub n_nextn_predict: u32,
}

impl SessionLedger {
    pub fn new(family: ModelFamily, backend: SessionBackend, ctx: i32, prefill_cap: u32) -> Self {
        Self {
            family,
            backend,
            ctx,
            prefill_cap: if prefill_cap == 0 { 1 } else { prefill_cap },
            tokens: Vec::new(),
            valid: false,
            generation: 1,
            solar_state_valid: family != ModelFamily::SolarOpen2,
            mtp_draft_valid: false,
            n_swa: 0,
            n_swa_period: 0,
            n_layer: 0,
            n_nextn_predict: 0,
        }
    }

    pub fn set_n_swa(&mut self, n_swa: u32) {
        self.n_swa = n_swa;
    }

    pub fn apply_shape(&mut self, shape: &Shape) {
        self.n_swa = shape.n_swa;
        self.n_swa_period = shape.n_swa_period;
        self.n_layer = shape.n_layer;
        self.n_nextn_predict = shape.n_nextn_predict;
    }

    pub fn pos(&self) -> i32 {
        self.tokens.len() as i32
    }

    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    pub fn live_len(&self) -> i32 {
        if self.valid {
            self.pos()
        } else {
            0
        }
    }

    pub fn rewrite_requires_rebuild(live_len: i32, canonical_len: i32, common: i32) -> bool {
        if live_len < 0 || canonical_len < 0 || common < 0 {
            return true;
        }
        if common > live_len || common > canonical_len {
            return true;
        }
        common < live_len
    }

    pub fn common_prefix(live: &[i32], prompt: &[i32]) -> i32 {
        let n = live.len().min(prompt.len());
        let mut i = 0;
        while i < n && live[i] == prompt[i] {
            i += 1;
        }
        i as i32
    }

    pub fn common_prefix_with(&self, prompt: &[i32]) -> i32 {
        if !self.valid {
            return 0;
        }
        Self::common_prefix(&self.tokens, prompt)
    }

    pub fn starts_with_checkpoint(&self, prompt: &[i32]) -> bool {
        if !self.valid || prompt.len() < self.tokens.len() {
            return false;
        }
        prompt[..self.tokens.len()] == self.tokens[..]
    }

    pub fn prompt_exceeds_context(&self, prompt_len: i32) -> bool {
        prompt_len <= 0 || prompt_len > self.ctx
    }

    pub fn rewrite_from_common(&self, prompt: &[i32], common: i32) -> RewriteKind {
        if self.prompt_exceeds_context(prompt.len() as i32) {
            return RewriteKind::Error;
        }
        if !self.valid {
            return RewriteKind::Error;
        }
        if common < 0 || common > self.pos() || common > prompt.len() as i32 {
            return RewriteKind::Error;
        }
        for i in 0..common as usize {
            if self.tokens[i] != prompt[i] {
                return RewriteKind::Error;
            }
        }
        if common == self.pos() {
            return RewriteKind::Extend;
        }
        if Self::rewrite_requires_rebuild(self.pos(), prompt.len() as i32, common) {
            return RewriteKind::Rebuild;
        }
        RewriteKind::Error
    }

    /// C `exaone_layer_is_sliding`: LLLG full attention on every
    /// `n_swa_period`-th layer, sliding otherwise.
    pub fn exaone_layer_is_sliding(il: u32, n_swa_period: u32) -> bool {
        n_swa_period != 0 && (il % n_swa_period) != n_swa_period - 1
    }

    /// C `exaone_graph_prefill_cap_for_context`.
    pub fn exaone_graph_prefill_cap_for_context(ctx_size: u32, requested: u32) -> u32 {
        let cap = if requested == 0 { 512 } else { requested };
        cap.min(ctx_size)
    }

    /// C `exaone_graph_layer_kv_cap`. Sliding rings keep the SWA window
    /// plus one prefill chunk. GPU alloc stays native; the number is host.
    pub fn exaone_graph_layer_kv_cap(
        il: u32,
        ctx_size: u32,
        prefill_cap: u32,
        n_swa: u32,
        n_swa_period: u32,
    ) -> u32 {
        if !Self::exaone_layer_is_sliding(il, n_swa_period) {
            return ctx_size;
        }
        (u64::from(n_swa) + u64::from(prefill_cap)).min(u64::from(ctx_size)) as u32
    }

    /// Planned sliding caps after a successful CUDA session create.
    /// `DS4_NO_GPU` / CPU backend stay 0, matching C.
    pub fn planned_exaone_rewind_span(&self) -> i32 {
        if self.family != ModelFamily::ExaoneMoe || self.backend != SessionBackend::Cuda {
            return 0;
        }
        if self.n_swa_period == 0 || self.n_layer == 0 {
            return 0;
        }
        let ctx = self.ctx.max(0) as u32;
        let prefill = Self::exaone_graph_prefill_cap_for_context(ctx, self.prefill_cap);
        let n_exec = self.n_layer.saturating_sub(self.n_nextn_predict);
        let caps: Vec<u32> = (0..n_exec)
            .filter(|&il| Self::exaone_layer_is_sliding(il, self.n_swa_period))
            .map(|il| {
                Self::exaone_graph_layer_kv_cap(il, ctx, prefill, self.n_swa, self.n_swa_period)
            })
            .collect();
        Self::exaone_rewind_span(self.family, true, ctx, self.live_len(), self.n_swa, &caps)
    }

    /// C `ds4_session_exaone_rewind_span` arithmetic. `sliding_caps` are the
    /// SWA layer KV capacities that have a live tensor. GPU graph alloc
    /// stays native; this is the host policy once those caps are known.
    pub fn exaone_rewind_span(
        family: ModelFamily,
        graph_ready: bool,
        ctx_size: u32,
        live: i32,
        n_swa: u32,
        sliding_caps: &[u32],
    ) -> i32 {
        if family != ModelFamily::ExaoneMoe || !graph_ready {
            return 0;
        }
        let mut narrowest = 0u32;
        let mut bounded = false;
        for &cap in sliding_caps {
            if !bounded || cap < narrowest {
                narrowest = cap;
                bounded = true;
            }
        }
        if !bounded {
            return ctx_size as i32;
        }
        if live <= 0 {
            return 0;
        }
        if (live as u32) <= narrowest {
            return live;
        }
        if narrowest <= n_swa {
            return 0;
        }
        (narrowest - n_swa + 1) as i32
    }

    pub fn exaone_sync_start(live: i32, prompt_len: i32, common: i32, span: i32) -> (i32, bool) {
        let mut start = common;
        if start > 0 && start == prompt_len {
            start = prompt_len - 1;
        }
        if live - start > span {
            start = 0;
        }
        let bump = common < live;
        (start, bump)
    }

    pub fn prefill_fence_rows() -> u32 {
        match std::env::var("DS4_PREFILL_NOFENCE") {
            Ok(v) if !v.is_empty() && v != "0" => 0,
            _ => PREFILL_FENCE_ROWS,
        }
    }

    pub fn plan_sync(&self, prompt: &[i32], exaone_span: i32) -> SyncPlan {
        let plen = prompt.len() as i32;
        if self.prompt_exceeds_context(plen) {
            return SyncPlan {
                err: true,
                start: 0,
                rebuild: false,
                bump: false,
                fence: false,
                bounds: true,
            };
        }

        if self.family == ModelFamily::ExaoneMoe {
            let live = self.live_len();
            let (start, bump) = if self.valid {
                let common = self.common_prefix_with(prompt);
                Self::exaone_sync_start(live, plen, common, exaone_span)
            } else {
                (0, false)
            };
            return SyncPlan {
                err: false,
                start,
                rebuild: start == 0,
                bump,
                fence: false,
                bounds: false,
            };
        }

        let solar_ok = self.family != ModelFamily::SolarOpen2 || self.solar_state_valid;
        let can_extend = self.starts_with_checkpoint(prompt) && solar_ok;
        if can_extend {
            let mut start = self.pos();
            if self.family == ModelFamily::Dots3Note && start > 0 && self.prefill_cap > 0 {
                let tail = (start as u32) % self.prefill_cap;
                if tail != 0 {
                    start -= tail as i32;
                }
            }
            return SyncPlan {
                err: false,
                start,
                rebuild: false,
                bump: false,
                fence: false,
                bounds: false,
            };
        }

        let bump = !matches!(self.family, ModelFamily::Motif3 | ModelFamily::Dots3Note);
        if self.family == ModelFamily::DeepSeek4 && self.backend == SessionBackend::Cuda {
            let f = Self::prefill_fence_rows();
            let mut width = self.prefill_cap;
            if (plen as u32) < width {
                width = plen as u32;
            }
            if f != 0 && width > f {
                return SyncPlan {
                    err: true,
                    start: 0,
                    rebuild: true,
                    bump: false,
                    fence: true,
                    bounds: false,
                };
            }
        }
        SyncPlan {
            err: false,
            start: 0,
            rebuild: true,
            bump,
            fence: false,
            bounds: false,
        }
    }

    pub fn commit_sync(&mut self, prompt: &[i32], plan: &SyncPlan) {
        if plan.err {
            return;
        }
        if plan.bump {
            self.generation += 1;
        }
        self.tokens = prompt.to_vec();
        self.valid = true;
        self.mtp_draft_valid = false;
        if self.family == ModelFamily::SolarOpen2 {
            self.solar_state_valid = true;
        }
    }

    pub fn commit_eval(&mut self, token: i32) {
        self.tokens.push(token);
        self.valid = true;
        self.mtp_draft_valid = false;
    }

    pub fn rewind(&mut self, mut pos: i32) -> RewindResult {
        let old = self.pos();
        if pos < 0 {
            pos = 0;
        }
        if pos > old {
            pos = old;
        }
        let bump = pos < old;
        if bump {
            self.generation += 1;
        }
        self.tokens.truncate(pos as usize);
        self.mtp_draft_valid = false;
        let solar = self.family == ModelFamily::SolarOpen2;
        let solar_invalid = solar && pos != old;
        if solar_invalid {
            self.valid = false;
            self.solar_state_valid = false;
        }
        // A recurrent block's history is its state, so a truncation cannot be
        // partial: the next sync replays the prefix from zero.
        if matches!(self.family, ModelFamily::Step37 | ModelFamily::Ling3Vl) && pos != old {
            self.valid = false;
        }
        RewindResult {
            pos,
            bump,
            solar_invalid,
            valid: self.valid,
            generation: self.generation,
        }
    }

    pub fn invalidate(&mut self) {
        self.generation += 1;
        self.valid = false;
        self.tokens.clear();
        self.mtp_draft_valid = false;
        if self.family == ModelFamily::SolarOpen2 {
            self.solar_state_valid = false;
        }
    }

    /// Replace the host token timeline from a parsed DSV4 prefix.
    /// Does not bump `generation`; native `ds4_session_load_payload` owns that.
    pub fn replace_checkpoint(&mut self, tokens: &[i32]) {
        self.tokens = tokens.to_vec();
        self.valid = true;
        self.mtp_draft_valid = false;
        if self.family == ModelFamily::SolarOpen2 {
            self.solar_state_valid = true;
        }
    }

    /// Native load failed after C already bumped generation and dropped KV.
    pub fn clear_checkpoint_keep_generation(&mut self) {
        self.tokens.clear();
        self.valid = false;
        self.mtp_draft_valid = false;
        if self.family == ModelFamily::SolarOpen2 {
            self.solar_state_valid = false;
        }
    }
}

pub fn dump_cmd(cmd: &str, args: &[&str]) -> String {
    match cmd {
        "rewrite" if args.len() == 3 => {
            let live: i32 = args[0].parse().unwrap_or(-1);
            let canon: i32 = args[1].parse().unwrap_or(-1);
            let common: i32 = args[2].parse().unwrap_or(-1);
            format!(
                "REBUILD {}\n",
                u32::from(SessionLedger::rewrite_requires_rebuild(live, canon, common))
            )
        }
        "prefix" if args.len() == 2 => {
            let live = parse_ids(args[0]);
            let prompt = parse_ids(args[1]);
            format!("COMMON {}\n", SessionLedger::common_prefix(&live, &prompt))
        }
        "rewrite-from" if args.len() == 4 => {
            let ctx: i32 = args[0].parse().unwrap_or(0);
            let live = parse_ids(args[1]);
            let prompt = parse_ids(args[2]);
            let common: i32 = args[3].parse().unwrap_or(-1);
            let mut host =
                SessionLedger::new(ModelFamily::DeepSeek4, SessionBackend::Cuda, ctx, 64);
            host.tokens = live;
            host.valid = true;
            let kind = host.rewrite_from_common(&prompt, common);
            let tag = match kind {
                RewriteKind::Error => "error",
                RewriteKind::Extend => "extend",
                RewriteKind::Rebuild => "rebuild",
            };
            format!("REWRITE {tag}\n")
        }
        "plan" if args.len() == 8 => {
            let family = ModelFamily::from_oracle_name(args[0]).unwrap_or(ModelFamily::DeepSeek4);
            let backend = SessionBackend::from_oracle_name(args[1]).unwrap_or(SessionBackend::Cuda);
            let ctx: i32 = args[2].parse().unwrap_or(0);
            let prefill: u32 = args[3].parse().unwrap_or(1);
            let valid = args[4] != "0";
            let solar = args[5] != "0";
            let span: i32 = args[6].parse().unwrap_or(0);
            let rest = args[7];
            let (live, prompt) = split_pair(rest);
            let mut host = SessionLedger::new(family, backend, ctx, prefill);
            host.tokens = live;
            host.valid = valid;
            host.solar_state_valid = solar;
            let plan = host.plan_sync(&prompt, span);
            format!(
                "PLAN err={} start={} rebuild={} bump={} fence={} bounds={}\n",
                u32::from(plan.err),
                plan.start,
                u32::from(plan.rebuild),
                u32::from(plan.bump),
                u32::from(plan.fence),
                u32::from(plan.bounds)
            )
        }
        "rewind" if args.len() == 4 => {
            let ctx: i32 = args[0].parse().unwrap_or(8);
            let live: i32 = args[1].parse().unwrap_or(0);
            let pos: i32 = args[2].parse().unwrap_or(0);
            let solar = args[3] != "0";
            let family = if solar {
                ModelFamily::SolarOpen2
            } else {
                ModelFamily::DeepSeek4
            };
            let mut host = SessionLedger::new(family, SessionBackend::Cuda, ctx, 64);
            host.tokens = (0..live).collect();
            host.valid = true;
            host.solar_state_valid = true;
            let r = host.rewind(pos);
            format!(
                "REWIND pos={} bump={} solar_invalid={} valid={} gen={}\n",
                r.pos,
                u32::from(r.bump),
                u32::from(r.solar_invalid),
                u32::from(r.valid),
                r.generation
            )
        }
        "invalidate" => {
            let mut host = SessionLedger::new(ModelFamily::DeepSeek4, SessionBackend::Cuda, 16, 64);
            host.tokens = vec![1, 2, 3];
            host.valid = true;
            host.invalidate();
            format!(
                "INVALID gen={} valid={} len={}\n",
                host.generation,
                u32::from(host.valid),
                host.pos()
            )
        }
        "create" if args.len() == 1 => {
            let ctx: i32 = args[0].parse().unwrap_or(0);
            let host = SessionLedger::new(ModelFamily::DeepSeek4, SessionBackend::Cuda, ctx, 64);
            format!(
                "CREATE gen={} pos={} valid={} ctx={}\n",
                host.generation,
                host.pos(),
                u32::from(host.valid),
                host.ctx
            )
        }
        _ => "ERROR unknown-cmd\n".into(),
    }
}

fn parse_ids(s: &str) -> Vec<i32> {
    if s.is_empty() || s == "-" {
        return Vec::new();
    }
    if let Some(rest) = s.strip_prefix("n:") {
        let n: usize = rest.parse().unwrap_or(0);
        return vec![1; n];
    }
    s.split(',').filter_map(|p| p.parse().ok()).collect()
}

fn split_pair(s: &str) -> (Vec<i32>, Vec<i32>) {
    let mut parts = s.splitn(2, ';');
    let a = parts.next().unwrap_or("");
    let b = parts.next().unwrap_or("");
    (parse_ids(a), parse_ids(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_rewind_requires_logits_replay() {
        let mut host = SessionLedger::new(ModelFamily::Step37, SessionBackend::Cuda, 1024, 64);
        let prompt = [1, 2, 3, 4];
        let plan = host.plan_sync(&prompt, 0);
        host.commit_sync(&prompt, &plan);
        let generation = host.generation;
        assert!(host.rewind(4).valid);
        assert_eq!(host.generation, generation);
        assert!(!host.rewind(3).valid);
        let replay = host.plan_sync(&prompt, 0);
        assert!(replay.rebuild);
        assert_eq!(replay.start, 0);
    }

    #[test]
    fn allows_prompt_that_exactly_fills_context() {
        let host = SessionLedger::new(ModelFamily::Qwen4Exp, SessionBackend::Cuda, 8, 8);

        let exact = host.plan_sync(&[1; 8], 0);
        assert!(!exact.err);
        assert!(!exact.bounds);

        let oversized = host.plan_sync(&[1; 9], 0);
        assert!(oversized.err);
        assert!(oversized.bounds);
    }

    #[test]
    fn exaone_rewind_span_matches_c_kernel_fixture() {
        let short = SessionLedger::exaone_rewind_span(
            ModelFamily::ExaoneMoe,
            true,
            262144,
            500,
            128,
            &[640],
        );
        let wrapped = SessionLedger::exaone_rewind_span(
            ModelFamily::ExaoneMoe,
            true,
            262144,
            1000,
            128,
            &[640],
        );
        assert_eq!((short, wrapped), (500, 513));
        assert_eq!(
            SessionLedger::exaone_rewind_span(
                ModelFamily::DeepSeek4,
                true,
                262144,
                500,
                128,
                &[640]
            ),
            0
        );
        assert_eq!(
            SessionLedger::exaone_rewind_span(
                ModelFamily::ExaoneMoe,
                false,
                262144,
                500,
                128,
                &[640]
            ),
            0
        );
        assert_eq!(
            SessionLedger::exaone_rewind_span(ModelFamily::ExaoneMoe, true, 262144, 1000, 128, &[]),
            262144
        );
        assert_eq!(
            SessionLedger::exaone_rewind_span(ModelFamily::ExaoneMoe, true, 262144, 0, 128, &[640]),
            0
        );
        assert_eq!(
            SessionLedger::exaone_rewind_span(
                ModelFamily::ExaoneMoe,
                true,
                262144,
                1000,
                128,
                &[128]
            ),
            0
        );
        assert_eq!(
            SessionLedger::exaone_rewind_span(
                ModelFamily::ExaoneMoe,
                true,
                262144,
                1000,
                128,
                &[900, 640, 800]
            ),
            513
        );
    }

    #[test]
    fn exaone_layer_kv_cap_matches_c_kernel_fixture() {
        use crate::shape::SHAPE_KEXAONE_236B;
        let n_swa = SHAPE_KEXAONE_236B.n_swa;
        let period = SHAPE_KEXAONE_236B.n_swa_period;
        assert_eq!(
            SessionLedger::exaone_graph_layer_kv_cap(0, 262144, 512, n_swa, period),
            640
        );
        assert_eq!(
            SessionLedger::exaone_graph_layer_kv_cap(3, 262144, 512, n_swa, period),
            262144
        );
        assert!(SessionLedger::exaone_layer_is_sliding(0, period));
        assert!(!SessionLedger::exaone_layer_is_sliding(3, period));
        assert_eq!(
            SessionLedger::exaone_graph_prefill_cap_for_context(262144, 0),
            512
        );
    }

    #[test]
    fn planned_exaone_rewind_span_uses_host_caps() {
        use crate::shape::SHAPE_KEXAONE_236B;
        let mut host =
            SessionLedger::new(ModelFamily::ExaoneMoe, SessionBackend::Cuda, 262144, 512);
        host.apply_shape(&SHAPE_KEXAONE_236B);
        host.tokens = (0..500).collect();
        host.valid = true;
        assert_eq!(host.planned_exaone_rewind_span(), 500);
        host.tokens = (0..1000).collect();
        assert_eq!(host.planned_exaone_rewind_span(), 513);
        host.backend = SessionBackend::Cpu;
        assert_eq!(host.planned_exaone_rewind_span(), 0);
    }
}
