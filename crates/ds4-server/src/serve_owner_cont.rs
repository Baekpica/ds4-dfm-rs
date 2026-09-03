use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex};

use super::*;
use crate::route::{route_decide, RouteEnv, LANE_CONTINUOUS};
use crate::serve_cont::{cont_prompt_tokens, ContExec, ContPair, ContWork};
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
    // A one-bank executor cannot overlap a sibling. Leave the FIFO untouched
    // so the owner runs the next request after this generation completes.
    if exec.max_seq() < 2 {
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return None;
    }
    let second = match jobs_rx.try_recv() {
        Ok(next) => Some(next),
        Err(TryRecvError::Disconnected) => None,
        Err(TryRecvError::Empty) => {
            let wait = super::owner_static::coalesce_wait_from_env();
            (!wait.is_zero())
                .then(|| jobs_rx.recv_timeout(wait).ok())
                .flatten()
        }
    };
    let Some(second) = second else {
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return None;
    };
    if cfg.max_queue_age_s > 0.0
        && second.prepared.arrived_at.elapsed().as_secs_f64() > cfg.max_queue_age_s
    {
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return Some(second);
    }
    if second.sink.state.gone() || second.sink.state.observe_disconnect() {
        finish_canceled_roll(second);
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return None;
    }
    let second_len = cont_prompt_tokens(exec, &second.prepared.parsed)
        .map(|(_, toks)| toks.len() as i32)
        .unwrap_or(0);
    let second_env = roll_route_env(cfg, exec, second_len);
    let second_dec = route_decide(
        second.prepared.parsed.needs,
        second.prepared.surface,
        &second_env,
    );
    if second_dec.lane != LANE_CONTINUOUS {
        run_owner_job(cfg, inner, engine, Some(exec), job);
        return Some(second);
    }
    serve_pair(cfg, inner, engine, exec, job, second);
    None
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

fn serve_pair(
    cfg: &ServerConfig,
    inner: &Arc<Mutex<ServerInner>>,
    engine: &mut dyn DecodeIo,
    exec: &mut dyn ContExec,
    mut first: OwnerJob,
    mut second: OwnerJob,
) {
    first.lease.start();
    second.lease.start();
    let id_a = next_job_id(&mut lock_inner(inner).admit, first.prepared.parsed.kind);
    let id_b = next_job_id(&mut lock_inner(inner).admit, second.prepared.parsed.kind);
    let mut bank_hold_retry = |bank, live| {
        lock_inner(inner)
            .creg
            .bank_hold_retry(bank, live, monotonic_now())
    };
    let store = engine.kv_store_mut();
    let arrived_a = first.prepared.arrived_at;
    let arrived_b = second.prepared.arrived_at;
    let decode_remaining = u32::try_from(first.prepared.parsed.max_tokens.max(1)).unwrap_or(1);
    let prefill_remaining = cont_prompt_tokens(exec, &second.prepared.parsed)
        .ok()
        .and_then(|(_, toks)| u32::try_from(toks.len()).ok())
        .unwrap_or(1)
        .max(1);
    let _ops = owner_tick_pair(
        PrefillChunkPolicy::from_env(),
        decode_remaining,
        prefill_remaining,
    );
    let [result_a, result_b] = exec.generate_pair(
        ContPair {
            first: ContWork {
                parsed: &first.prepared.parsed,
                job_id: &id_a,
                created: unix_now(),
                cors: cfg.cors,
                default_tokens: cfg.default_tokens,
                t_arrive: arrived_a,
                out: &mut first.sink,
            },
            second: ContWork {
                parsed: &second.prepared.parsed,
                job_id: &id_b,
                created: unix_now(),
                cors: cfg.cors,
                default_tokens: cfg.default_tokens,
                t_arrive: arrived_b,
                out: &mut second.sink,
            },
        },
        &mut bank_hold_retry,
        store,
    );
    let served = result_a.is_ok() as usize + result_b.is_ok() as usize;
    let fallback = matches!(&result_a, Err(GenerateError::Unsupported(_))) as usize
        + matches!(&result_b, Err(GenerateError::Unsupported(_))) as usize;
    eprintln!("ds4-server-rs: continuous batch path=cont served={served} fallback={fallback}");
    settle_roll_job(cfg, inner, engine, first, &id_a, result_a);
    settle_roll_job(cfg, inner, engine, second, &id_b, result_b);
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
    let mut settlement = if job.sink.state.gone() {
        Settlement::CANCELED
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
