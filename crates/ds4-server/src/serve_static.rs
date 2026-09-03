//! Static-lane owner around `BatchCtx::generate_static` (C `n>=2`).

use crate::generate::GenerateError;

#[path = "serve_static_coalesce.rs"]
mod coalesce;
pub use coalesce::{
    coalesce_take, job_tok_footprint, static_ctx_overflow, static_peer_ok, CoalesceLimits,
    CoalescePeer, StaticPeerSpec, COALESCE_HARD_MAX,
};

/// C `ds4_bridge_batch_ctx_generate_static` when `n` is not a legal width.
pub const STATIC_WIDTH_ERR: &str = "static batch request count is out of range";

/// C `generate_batch_jobs` when the batched `err` buffer is empty.
pub const STATIC_FALLBACK_ERR: &str = "out of memory";

/// C `generate_batch_jobs` admission: coalesced group only.
pub const STATIC_N_MIN: usize = 2;

/// One greedy static row. Tokens are borrowed only for the call.
#[derive(Clone, Copy)]
pub struct StaticJob<'a> {
    pub tokens: &'a [i32],
    pub max_new_tokens: i32,
    pub eos: i32,
}

/// Owned sibling waiting to coalesce with the next static-routed request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedStaticJob {
    pub tokens: Vec<i32>,
    pub max_new_tokens: i32,
    pub eos: i32,
}

/// C `ds4_batch_gen_result.finish`: 1 = EOS → `"stop"`, 0 = budget → `"length"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticFinish {
    Length,
    Stop,
}

impl StaticFinish {
    pub const fn reason(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }
}

/// C `write_batch_completion`: clamp to decode budget and force `"length"`.
pub fn settle_static_finish(
    finish: StaticFinish,
    n_tokens: usize,
    budget: i32,
) -> (&'static str, usize) {
    let budget = budget.max(0) as usize;
    if n_tokens > budget {
        ("length", budget)
    } else {
        (finish.reason(), n_tokens)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticRow {
    pub tokens: Vec<i32>,
    pub finish: StaticFinish,
}

/// Trait seam so tests can spy on `generate_static` without a GGUF.
pub trait StaticExec {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError>;

    /// Extra rows already waiting (coalesce). Default: none.
    fn pending_siblings(&self) -> &[OwnedStaticJob] {
        &[]
    }

    fn coalesce_limits(&self) -> CoalesceLimits {
        CoalesceLimits::UNBOUNDED
    }

    fn ctx_max_seq(&self) -> i32 {
        i32::MAX
    }

    fn ctx_max_tokens(&self) -> i32 {
        i32::MAX
    }

    /// C per-call fallback after `generate_static` fails. Default: C err text.
    fn fallback_static(&mut self, err: GenerateError) -> Result<Vec<StaticRow>, GenerateError> {
        Err(static_fallback_error(err))
    }
}

/// Map a failed `generate_static` onto C `generate_batch_jobs` err text.
pub fn static_fallback_error(err: GenerateError) -> GenerateError {
    match err {
        GenerateError::Engine(msg) if !msg.is_empty() => GenerateError::Engine(msg),
        GenerateError::Engine(_)
        | GenerateError::Unsupported(_)
        | GenerateError::Streamed(_)
        | GenerateError::Io
        | GenerateError::ContinuationHold { .. } => {
            GenerateError::Engine(STATIC_FALLBACK_ERR.to_string())
        }
    }
}

/// Production owner: `BatchCtx::generate_static` with no GGUF in this crate.
#[cfg(feature = "native")]
pub struct BatchStatic<'a, 'm> {
    ctx: &'a mut ds4_core::BatchCtx<'m>,
}

#[cfg(feature = "native")]
impl<'a, 'm> BatchStatic<'a, 'm> {
    pub fn new(ctx: &'a mut ds4_core::BatchCtx<'m>) -> Self {
        Self { ctx }
    }
}

#[cfg(feature = "native")]
impl StaticExec for BatchStatic<'_, '_> {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError> {
        let requests: Vec<ds4_core::StaticBatchRequest<'_>> = jobs
            .iter()
            .map(|job| ds4_core::StaticBatchRequest {
                tokens: job.tokens,
                max_new_tokens: job.max_new_tokens,
                eos: job.eos,
            })
            .collect();
        self.ctx
            .generate_static(&requests)
            .map(|rows| {
                rows.into_iter()
                    .map(|row| StaticRow {
                        tokens: row.tokens,
                        finish: match row.finish {
                            ds4_core::StaticBatchFinish::Budget => StaticFinish::Length,
                            ds4_core::StaticBatchFinish::Eos => StaticFinish::Stop,
                        },
                    })
                    .collect()
            })
            .map_err(|err| GenerateError::Engine(err.message))
    }

    fn ctx_max_seq(&self) -> i32 {
        self.ctx.max_seq()
    }

    fn coalesce_limits(&self) -> CoalesceLimits {
        CoalesceLimits {
            cap: self.ctx.max_seq().max(1) as usize,
            max_tok_total: 0,
        }
    }
}

/// Used when the route is static but no owner is attached (n<2 still refuses).
pub struct DetachedStatic;

impl StaticExec for DetachedStatic {
    fn generate_static(
        &mut self,
        _jobs: &[StaticJob<'_>],
    ) -> Result<Vec<StaticRow>, GenerateError> {
        Err(GenerateError::Unsupported("static owner is not attached"))
    }
}

/// `None` means `n` is admitted at the owner boundary.
pub const fn static_width_error(n: usize) -> Option<&'static str> {
    if n < STATIC_N_MIN {
        Some(STATIC_WIDTH_ERR)
    } else {
        None
    }
}

/// Owner entry: refuse `n<2` with the C width string; otherwise call
/// [`StaticExec::generate_static`]. On err, C fallback text — never serial.
pub fn run_static(
    exec: &mut dyn StaticExec,
    jobs: &[StaticJob<'_>],
) -> Result<Vec<StaticRow>, GenerateError> {
    if let Some(msg) = static_width_error(jobs.len()) {
        return Err(GenerateError::Engine(msg.to_string()));
    }
    let packed: i64 = jobs.iter().map(|job| job.tokens.len() as i64).sum();
    if static_ctx_overflow(
        jobs.len(),
        packed,
        exec.ctx_max_seq(),
        exec.ctx_max_tokens(),
    ) {
        return exec.fallback_static(GenerateError::Engine(STATIC_FALLBACK_ERR.to_string()));
    }
    match exec.generate_static(jobs) {
        Ok(rows) => Ok(rows),
        Err(err) => exec.fallback_static(err),
    }
}

/// Routed owner: current request plus C-gathered [`StaticExec::pending_siblings`].
pub fn run_static_routed(
    exec: &mut dyn StaticExec,
    current: StaticJob<'_>,
) -> Result<Vec<StaticRow>, GenerateError> {
    let siblings = exec.pending_siblings().to_vec();
    let limits = exec.coalesce_limits();
    let peers: Vec<CoalescePeer> = siblings
        .iter()
        .map(|sibling| CoalescePeer {
            footprint: job_tok_footprint(sibling.tokens.len(), sibling.max_new_tokens),
            peer_ok: true,
        })
        .collect();
    let take = coalesce_take(
        job_tok_footprint(current.tokens.len(), current.max_new_tokens),
        &peers,
        limits,
    );
    let mut jobs = Vec::with_capacity(take + 1);
    for sibling in siblings.iter().take(take) {
        jobs.push(StaticJob {
            tokens: &sibling.tokens,
            max_new_tokens: sibling.max_new_tokens,
            eos: sibling.eos,
        });
    }
    jobs.push(current);
    run_static(exec, &jobs)
}

#[path = "serve_static_settle.rs"]
mod settle;
pub use settle::{write_static_completion, StaticSettle};

#[cfg(test)]
#[path = "serve_static_harness.rs"]
mod harness;

#[cfg(test)]
#[path = "serve_static_test.rs"]
mod tests;

#[cfg(test)]
#[path = "serve_static_fallback_test.rs"]
mod fallback_tests;
