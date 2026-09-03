use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::*;
use crate::route::{route_decide, RouteEnv, LANE_CONTINUOUS};
use crate::serve_cont::{cont_prompt_tokens, ContExec, ContWork};
use crate::serve_cont_prefill::{owner_tick_pair, PrefillChunkPolicy};

pub(super) fn run_owner_maybe_roll(
    cfg: &ServerConfig,
    inner: &Arc<Mutex<ServerInner>>,
    engine: &mut dyn DecodeIo,
    exec: &mut dyn ContExec,
    job: OwnerJob,
    jobs_rx: &Receiver<OwnerJob>,
) -> Option<OwnerJob> {
    let prompt_len = cont_prompt_tokens(exec, &job.prepared.parsed)
        .map(|(_, toks)| toks.len() as i32)
        .unwrap_or(0);
    let env = roll_route_env(cfg, exec, prompt_len);
    let dec = route_decide(job.prepared.parsed.needs, job.prepared.surface, &env);
    if dec.lane != LANE_CONTINUOUS {
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return None;
    }
    let mut job = job;
    if !resolve_bank_continuation(inner, &mut job.prepared, exec) {
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return None;
    }
    let cap = usize::try_from(exec.max_seq().max(1)).unwrap_or(1);
    if cap == 1 {
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return None;
    }
    let wait = super::owner_static::coalesce_wait_from_env();
    let deadline = (!wait.is_zero()).then(|| Instant::now() + wait);
    let mut jobs = vec![job];
    let mut lookahead = None;
    while jobs.len() < cap {
        let next = match jobs_rx.try_recv() {
            Ok(job) => job,
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {
                let Some(deadline) = deadline else {
                    break;
                };
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                match jobs_rx.recv_timeout(deadline.saturating_duration_since(now)) {
                    Ok(job) => job,
                    Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
                }
            }
        };
        let mut next = next;
        if cfg.max_queue_age_s > 0.0
            && next.prepared.arrived_at.elapsed().as_secs_f64() > cfg.max_queue_age_s
        {
            lookahead = Some(next);
            break;
        }
        if next.sink.state.gone() || next.sink.state.observe_disconnect() {
            finish_canceled_roll(next);
            continue;
        }
        let prompt_len = cont_prompt_tokens(exec, &next.prepared.parsed)
            .map(|(_, toks)| toks.len() as i32)
            .unwrap_or(0);
        let env = roll_route_env(cfg, exec, prompt_len);
        let dec = route_decide(next.prepared.parsed.needs, next.prepared.surface, &env);
        if dec.lane != LANE_CONTINUOUS
            || !resolve_bank_continuation(inner, &mut next.prepared, exec)
            || next.prepared.parsed.directed_bank.is_some_and(|bank| {
                jobs.iter()
                    .any(|job| job.prepared.parsed.directed_bank == Some(bank))
            })
        {
            lookahead = Some(next);
            break;
        }
        jobs.push(next);
    }
    if jobs.len() == 1 {
        let job = jobs.pop().unwrap();
        run_owner_job(cfg, inner, engine, Some(exec), job);
    } else {
        serve_batch(cfg, inner, engine, exec, jobs);
    }
    lookahead
}

fn roll_route_env(cfg: &ServerConfig, exec: &dyn ContExec, prompt_len: i32) -> RouteEnv {
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

fn finish_canceled_roll(job: OwnerJob) {
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

fn serve_batch(
    cfg: &ServerConfig,
    inner: &Arc<Mutex<ServerInner>>,
    engine: &mut dyn DecodeIo,
    exec: &mut dyn ContExec,
    mut jobs: Vec<OwnerJob>,
) {
    for job in &mut jobs {
        job.lease.start();
    }
    let ids: Vec<_> = {
        let mut inner = lock_inner(inner);
        jobs.iter()
            .map(|job| next_job_id(&mut inner.admit, job.prepared.parsed.kind))
            .collect()
    };
    let mut bank_hold_retry = |bank, live| {
        lock_inner(inner)
            .creg
            .bank_hold_retry(bank, live, monotonic_now())
    };
    if let Some(second) = jobs.get(1) {
        let decode_remaining =
            u32::try_from(jobs[0].prepared.parsed.max_tokens.max(1)).unwrap_or(1);
        let prefill_remaining = cont_prompt_tokens(exec, &second.prepared.parsed)
            .ok()
            .and_then(|(_, toks)| u32::try_from(toks.len()).ok())
            .unwrap_or(1)
            .max(1);
        let _ = owner_tick_pair(
            PrefillChunkPolicy::from_env(),
            decode_remaining,
            prefill_remaining,
        );
    }
    let created = unix_now();
    let works = jobs
        .iter_mut()
        .zip(&ids)
        .map(|(job, id)| ContWork {
            parsed: &job.prepared.parsed,
            job_id: id,
            created,
            cors: cfg.cors,
            default_tokens: cfg.default_tokens,
            t_arrive: job.prepared.arrived_at,
            out: &mut job.sink,
        })
        .collect();
    let results = exec.generate_batch(works, &mut bank_hold_retry, engine.kv_store_mut());
    let served = results.iter().filter(|result| result.is_ok()).count();
    let fallback = results
        .iter()
        .filter(|result| matches!(result, Err(GenerateError::Unsupported(_))))
        .count();
    eprintln!("ds4-server-rs: continuous batch path=cont served={served} fallback={fallback}");
    let mut results = results.into_iter();
    for (job, id) in jobs.into_iter().zip(ids) {
        let result = results.next().unwrap_or_else(|| {
            Err(GenerateError::Engine(
                "continuous batch returned too few results".into(),
            ))
        });
        settle_roll_job(cfg, inner, engine, job, &id, result);
    }
}

fn settle_roll_job(
    cfg: &ServerConfig,
    inner: &Arc<Mutex<ServerInner>>,
    engine: &mut dyn DecodeIo,
    mut job: OwnerJob,
    id: &str,
    result: Result<GenerateOutcome, GenerateError>,
) {
    let arrived_at = job.prepared.arrived_at;
    if let Ok(outcome) = &result {
        publish_continuous_tool_turn(inner, job.prepared.parsed.api, outcome);
    }
    let mut settlement = if job.sink.state.gone() {
        Settlement::CANCELED
    } else if job.prepared.parsed.needs & NEED_BANK_FRONTIER != 0 {
        settle_bank_continuation(cfg, &job.prepared, result, &mut job.sink)
    } else if matches!(result, Err(GenerateError::Unsupported(_))) {
        run_serial(
            cfg,
            inner,
            &job.prepared,
            id,
            engine,
            None,
            &mut job.sink,
            arrived_at,
        )
    } else {
        settle_generation_result(cfg, &job.prepared, result, &mut job.sink)
    };
    if job.sink.state.slow() {
        settlement = settlement.slow_reader();
    }
    drop(job.sink);
    job.lease.settlement = settlement;
    lock_inner(inner).metrics.record_route(
        job.prepared.surface,
        LANE_CONTINUOUS,
        crate::route::REASON_CONT,
        job.prepared.parsed.think_mode,
    );
    if let Err(err) = job.done.send(job.lease) {
        let mut lease = err.0;
        lease.settlement = lease.settlement.transport_gone();
    }
}

#[cfg(test)]
#[path = "serve_owner_cont_test.rs"]
mod tests;
