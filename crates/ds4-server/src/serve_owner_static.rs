use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::*;
use crate::generate::GenerateError;
use crate::route::{route_decide, RouteEnv, LANE_STATIC};
use crate::serve_cont::{cont_prompt_tokens, ContExec};
use crate::serve_static::{
    coalesce_take, job_tok_footprint, run_static, static_peer_ok, CoalesceLimits, CoalescePeer,
    StaticJob, StaticPeerSpec, StaticRow,
};

struct Member {
    job: OwnerJob,
    tokens: Vec<i32>,
}

pub(super) fn run_owner_maybe_coalesce(
    cfg: &ServerConfig,
    inner: &Arc<Mutex<ServerInner>>,
    engine: &mut dyn DecodeIo,
    exec: &mut dyn ContExec,
    job: OwnerJob,
    jobs_rx: &Receiver<OwnerJob>,
) -> Option<OwnerJob> {
    if exec.as_static().is_none() {
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return None;
    }
    let tokens = cont_prompt_tokens(exec, &job.prepared.parsed)
        .map(|(_, toks)| toks)
        .unwrap_or_default();
    let env = static_route_env(cfg, exec, tokens.len() as i32);
    let dec = route_decide(job.prepared.parsed.needs, job.prepared.surface, &env);
    if dec.lane != LANE_STATIC {
        return super::owner_cont::run_owner_maybe_roll(cfg, inner, engine, exec, job, jobs_rx);
    }
    let limits = exec
        .as_static()
        .map(|owner| owner.coalesce_limits())
        .unwrap_or_default();
    let mut batch = vec![Member { job, tokens }];
    let lookahead = gather_peers(
        cfg,
        exec,
        &env,
        limits,
        &mut batch,
        jobs_rx,
        coalesce_wait_from_env(),
    );
    if batch.len() < 2 {
        let Member { job, .. } = batch.remove(0);
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return lookahead;
    }
    serve_group(cfg, inner, engine, exec, dec.reason, batch);
    lookahead
}

fn static_route_env(cfg: &ServerConfig, exec: &dyn ContExec, prompt_len: i32) -> RouteEnv {
    let (cont_tools_anthropic, cont_tools_responses) = process_cont_tools();
    RouteEnv {
        coalesce: true,
        have_cont: cfg.continuous,
        cont_anthropic: parse_default_on(std::env::var_os("DS4_SERVER_CONT_ANTHROPIC").as_deref()),
        cont_responses: parse_default_on(std::env::var_os("DS4_SERVER_CONT_RESPONSES").as_deref()),
        cont_tools_anthropic,
        cont_tools_responses,
        seq_cap: exec.seq_cap(),
        prompt_len,
    }
}

pub(super) fn coalesce_wait_from_env() -> Duration {
    // Same knob as C coalesce_gather: default 0 (drain only). A small
    // positive window lets two concurrent short HTTP jobs join n=2
    // instead of collapsing each to serial.
    let ms = std::env::var("DS4_SERVER_COALESCE_WAIT_MS")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(0)
        .max(0);
    Duration::from_millis(ms as u64)
}

fn gather_peers(
    cfg: &ServerConfig,
    exec: &dyn ContExec,
    env: &RouteEnv,
    limits: CoalesceLimits,
    batch: &mut Vec<Member>,
    jobs_rx: &Receiver<OwnerJob>,
    wait: Duration,
) -> Option<OwnerJob> {
    let deadline = (!wait.is_zero()).then(|| Instant::now() + wait);
    loop {
        if batch.len() >= limits.clamp().cap {
            return None;
        }
        let next = match jobs_rx.try_recv() {
            Ok(job) => job,
            Err(TryRecvError::Disconnected) => return None,
            Err(TryRecvError::Empty) => {
                let Some(deadline) = deadline else {
                    return None;
                };
                let now = Instant::now();
                if now >= deadline {
                    return None;
                }
                match jobs_rx.recv_timeout(deadline.saturating_duration_since(now)) {
                    Ok(job) => job,
                    Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => return None,
                }
            }
        };
        if cfg.max_queue_age_s > 0.0
            && next.prepared.arrived_at.elapsed().as_secs_f64() > cfg.max_queue_age_s
        {
            return Some(next);
        }
        if next.sink.state.gone() || next.sink.state.observe_disconnect() {
            finish_canceled(next);
            continue;
        }
        let tokens = cont_prompt_tokens(exec, &next.prepared.parsed)
            .map(|(_, toks)| toks)
            .unwrap_or_default();
        let peer_ok = static_peer_ok(StaticPeerSpec {
            needs: next.prepared.parsed.needs,
            surface: next.prepared.surface,
            cont_anthropic: env.cont_anthropic,
            cont_responses: env.cont_responses,
        });
        let tok_total: i64 = batch
            .iter()
            .map(|member| {
                job_tok_footprint(member.tokens.len(), member.job.prepared.parsed.max_tokens)
            })
            .sum();
        let take = coalesce_take(
            tok_total,
            &[CoalescePeer {
                footprint: job_tok_footprint(tokens.len(), next.prepared.parsed.max_tokens),
                peer_ok,
            }],
            limits,
        );
        if take == 1 {
            batch.push(Member { job: next, tokens });
        } else {
            return Some(next);
        }
    }
}

fn serve_group(
    cfg: &ServerConfig,
    inner: &Arc<Mutex<ServerInner>>,
    engine: &mut dyn DecodeIo,
    exec: &mut dyn ContExec,
    reason: u8,
    mut batch: Vec<Member>,
) {
    for member in &mut batch {
        member.job.lease.start();
    }
    let result = {
        let jobs: Vec<StaticJob<'_>> = batch
            .iter()
            .map(|member| StaticJob {
                tokens: &member.tokens,
                max_new_tokens: member.job.prepared.parsed.max_tokens,
                eos: -1,
            })
            .collect();
        match exec.as_static() {
            Some(owner) => run_static(owner, &jobs),
            None => Err(GenerateError::Unsupported("static owner is not attached")),
        }
    };
    for (index, member) in batch.into_iter().enumerate() {
        settle_member(cfg, inner, engine, reason, member, index, &result);
    }
}

fn settle_member(
    cfg: &ServerConfig,
    inner: &Arc<Mutex<ServerInner>>,
    engine: &dyn DecodeIo,
    reason: u8,
    member: Member,
    index: usize,
    result: &Result<Vec<StaticRow>, GenerateError>,
) {
    let Member {
        job:
            OwnerJob {
                prepared,
                mut sink,
                done,
                mut lease,
            },
        tokens,
    } = member;
    let id = next_job_id(&mut lock_inner(inner).admit, prepared.parsed.kind);
    lock_inner(inner).metrics.record_route(
        prepared.surface,
        LANE_STATIC,
        reason,
        prepared.parsed.think_mode,
    );
    let one = match result {
        Ok(rows) => Ok(rows.get(index).cloned().into_iter().collect()),
        Err(GenerateError::Engine(msg)) => Err(GenerateError::Engine(msg.clone())),
        Err(GenerateError::Unsupported(msg)) => Err(GenerateError::Unsupported(*msg)),
        Err(GenerateError::Streamed(msg)) => Err(GenerateError::Streamed(msg.clone())),
        Err(GenerateError::Io) => Err(GenerateError::Io),
        Err(GenerateError::ContinuationHold { retry_after }) => {
            Err(GenerateError::ContinuationHold {
                retry_after: *retry_after,
            })
        }
    };
    let mut settlement = settle_static_lane(
        cfg,
        &prepared,
        &id,
        engine,
        i32::try_from(tokens.len()).unwrap_or(i32::MAX),
        one,
        &mut sink,
    );
    if sink.state.slow() {
        settlement = settlement.slow_reader();
    }
    drop(sink);
    lease.settlement = settlement;
    if let Err(err) = done.send(lease) {
        let mut lease = err.0;
        lease.settlement = lease.settlement.transport_gone();
    }
}

fn finish_canceled(job: OwnerJob) {
    let OwnerJob {
        sink,
        done,
        mut lease,
        ..
    } = job;
    lease.start();
    drop(sink);
    lease.settlement = Settlement::CANCELED;
    if let Err(err) = done.send(lease) {
        let mut lease = err.0;
        lease.settlement = lease.settlement.transport_gone();
    }
}

#[cfg(test)]
#[path = "serve_owner_static_test.rs"]
mod tests;
