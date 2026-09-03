use std::cell::Cell;
use std::io::Write;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::super::*;
use super::run_owner_maybe_coalesce;
use crate::generate::{GenerateError, GenerateOutcome, ScriptedDecode};
use crate::parse::ParseEnv;
use crate::route::{ThinkMode, WireSurface, LANE_SERIAL, LANE_STATIC};
use crate::serve_cont::ContExec;
use crate::serve_static::{
    static_fallback_error, CoalesceLimits, StaticExec, StaticFinish, StaticJob, StaticRow,
};

#[derive(Default)]
struct OwnerSpy {
    calls: usize,
    ns: Vec<usize>,
    seq_cap: i32,
    encode_calls: Cell<usize>,
    allow_cont: bool,
    cont_calls: usize,
}

impl StaticExec for OwnerSpy {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError> {
        self.calls += 1;
        self.ns.push(jobs.len());
        Ok(jobs
            .iter()
            .map(|job| StaticRow {
                tokens: job.tokens.to_vec(),
                finish: StaticFinish::Stop,
            })
            .collect())
    }

    fn fallback_static(&mut self, err: GenerateError) -> Result<Vec<StaticRow>, GenerateError> {
        Err(static_fallback_error(err))
    }
}

impl ContExec for OwnerSpy {
    fn model_id(&self) -> i32 {
        0
    }
    fn seq_cap(&self) -> i32 {
        self.seq_cap
    }
    fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
        self.encode_calls.set(self.encode_calls.get() + 1);
        vec![77]
    }
    fn encode_text(&self, _text: &str) -> Vec<i32> {
        self.encode_calls.set(self.encode_calls.get() + 1);
        vec![77]
    }
    fn generate(
        &mut self,
        _parsed: &crate::parse::ParsedRequest,
        _job_id: &str,
        _created: i64,
        _cors: bool,
        _default_tokens: i32,
        _t_arrive: Instant,
        _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut ds4_kv::Store>,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        assert!(self.allow_cont, "continuous generate on a static request");
        self.cont_calls += 1;
        out.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}")
            .map_err(|_| GenerateError::Io)?;
        Ok(GenerateOutcome {
            finish: "stop".into(),
            ..GenerateOutcome::default()
        })
    }
    fn as_static(&mut self) -> Option<&mut dyn StaticExec> {
        Some(self)
    }
}

#[test]
fn continuous_owner_tokenizes_a_long_request_once_before_generation() {
    let cfg = ServerConfig {
        continuous: true,
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job, drain) = static_chat_job(&inner, "long-session");
    let (_tx, rx) = mpsc::channel();
    let mut spy = OwnerSpy {
        seq_cap: 8192,
        allow_cont: true,
        ..OwnerSpy::default()
    };
    let mut engine = ScriptedDecode::from_pieces(&[]);

    assert!(run_owner_maybe_coalesce(&cfg, &inner, &mut engine, &mut spy, job, &rx).is_none());

    assert_eq!(spy.cont_calls, 1);
    assert_eq!(spy.encode_calls.get(), 1);
    let _ = drain.done.recv();
}

fn static_chat_job(inner: &Arc<Mutex<ServerInner>>, tag: &str) -> (OwnerJob, JobDrain) {
    let env = ParseEnv {
        default_model: "ds4".into(),
        default_tokens: 8,
        default_effort: ThinkMode::None,
        default_temp: 0.0,
        live_ids: Vec::new(),
    };
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":"{tag}"}}],"thinking":{{"type":"disabled"}},"temperature":0}}"#
    );
    let parsed = crate::parse::parse_request(WireSurface::OpenaiChat, &env, &body).unwrap();
    let prepared = PreparedJob {
        parsed,
        cont_prompt: None,
        surface: WireSurface::OpenaiChat,
        body_bytes: body.len() as u64,
        arrived_at: Instant::now(),
    };
    let mut g = lock_inner(inner);
    assert_eq!(enqueue(&mut g.admit, prepared.body_bytes), EnqVerdict::Ok);
    g.runtime.requests_started += 1;
    g.runtime.requests_inflight += 1;
    drop(g);
    let lease = JobLease::new(Arc::clone(inner), prepared.body_bytes, None);
    owner_job(prepared, lease)
}

#[test]
fn late_sibling_joins_when_coalesce_wait_is_set() {
    let cfg = ServerConfig {
        continuous: false,
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = static_chat_job(&inner, "a");
    let (job_b, drain_b) = static_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(30));
        tx.send(job_b).unwrap();
    });
    let mut spy = OwnerSpy::default();
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);
    std::env::set_var("DS4_SERVER_COALESCE_WAIT_MS", "100");
    let leftover = run_owner_maybe_coalesce(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);
    std::env::remove_var("DS4_SERVER_COALESCE_WAIT_MS");

    assert!(leftover.is_none());
    assert_eq!(spy.ns, vec![2]);
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
}

#[test]
fn two_queued_static_jobs_coalesce_on_owner_fifo() {
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = static_chat_job(&inner, "a");
    let (job_b, drain_b) = static_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    tx.send(job_b).unwrap();
    drop(tx);
    let mut spy = OwnerSpy::default();
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);

    let leftover = run_owner_maybe_coalesce(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);

    assert!(leftover.is_none());
    assert_eq!(spy.calls, 1);
    assert_eq!(spy.ns, vec![2]);
    let text_a = String::from_utf8(drain_a.state.take()).unwrap();
    let text_b = String::from_utf8(drain_b.state.take()).unwrap();
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
    assert!(text_a.starts_with("HTTP/1.1 200 OK"), "{text_a}");
    assert!(text_b.starts_with("HTTP/1.1 200 OK"), "{text_b}");
    assert!(!text_a.contains("serial-fallback"), "{text_a}");
    assert!(!text_b.contains("serial-fallback"), "{text_b}");
    let g = lock_inner(&inner);
    assert_eq!(g.runtime.requests_serial, 0);
    assert_eq!(
        g.metrics.route_requests[WireSurface::OpenaiChat as usize][LANE_STATIC as usize],
        2
    );
}

#[test]
fn continuous_zero_keeps_the_batch_ctx_for_two_short_static_jobs() {
    let cfg = ServerConfig {
        continuous: false,
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = static_chat_job(&inner, "a");
    let (job_b, drain_b) = static_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    tx.send(job_b).unwrap();
    drop(tx);
    let mut spy = OwnerSpy {
        seq_cap: 8,
        ..OwnerSpy::default()
    };
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);

    let leftover = run_owner_maybe_coalesce(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);

    assert!(leftover.is_none());
    assert_eq!(spy.calls, 1);
    assert_eq!(spy.ns, vec![2]);
    let text_a = String::from_utf8(drain_a.state.take()).unwrap();
    let text_b = String::from_utf8(drain_b.state.take()).unwrap();
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
    assert!(text_a.starts_with("HTTP/1.1 200 OK"), "{text_a}");
    assert!(text_b.starts_with("HTTP/1.1 200 OK"), "{text_b}");
    assert!(!text_a.contains("serial-fallback"), "{text_a}");
    assert!(!text_b.contains("serial-fallback"), "{text_b}");
    let g = lock_inner(&inner);
    assert_eq!(g.runtime.requests_serial, 0);
    assert_eq!(
        g.metrics.route_requests[WireSurface::OpenaiChat as usize][LANE_STATIC as usize],
        2
    );
}

#[test]
fn continuous_zero_n1_collapses_static_gather_to_serial() {
    // Given: CONTINUOUS=0 so route_decide picks STATIC; owner FIFO has one job
    let cfg = ServerConfig {
        continuous: false,
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = static_chat_job(&inner, "a");
    let (_tx, rx) = mpsc::channel();
    let mut spy = OwnerSpy {
        seq_cap: 8,
        ..OwnerSpy::default()
    };
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);

    // When: coalesce_gather width is 1
    let leftover = run_owner_maybe_coalesce(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);

    // Then: C worker_main n==1 → run_job_single. generate_static stays cold.
    assert!(leftover.is_none());
    assert_eq!(spy.calls, 0);
    let text = String::from_utf8(drain_a.state.take()).unwrap();
    let _ = drain_a.done.recv();
    assert!(text.starts_with("HTTP/1.1 200 OK"), "{text}");
    assert!(
        !text.contains(crate::serve_static::STATIC_WIDTH_ERR),
        "HTTP n=1 must not refuse width: {text}"
    );
    assert!(text.contains("serial-fallback"), "{text}");
    let g = lock_inner(&inner);
    assert_eq!(g.runtime.requests_serial, 1);
    assert_eq!(
        g.metrics.route_requests[WireSurface::OpenaiChat as usize][LANE_SERIAL as usize],
        1
    );
    assert_eq!(
        g.metrics.route_requests[WireSurface::OpenaiChat as usize][LANE_STATIC as usize],
        0
    );
}

#[test]
fn owner_ctx_overflow_uses_c_fallback_not_serial() {
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = static_chat_job(&inner, "a");
    let (job_b, drain_b) = static_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    tx.send(job_b).unwrap();
    drop(tx);
    let mut spy = OverflowSpy {
        inner: OwnerSpy::default(),
    };
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);

    let leftover = run_owner_maybe_coalesce(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);

    assert!(leftover.is_none());
    assert_eq!(spy.inner.calls, 0);
    let text_a = String::from_utf8(drain_a.state.take()).unwrap();
    let text_b = String::from_utf8(drain_b.state.take()).unwrap();
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
    assert!(text_a.contains("out of memory"), "{text_a}");
    assert!(text_b.contains("out of memory"), "{text_b}");
    assert!(!text_a.contains("serial-fallback"), "{text_a}");
    assert_eq!(lock_inner(&inner).runtime.requests_serial, 0);
}

struct OverflowSpy {
    inner: OwnerSpy,
}

impl StaticExec for OverflowSpy {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError> {
        self.inner.generate_static(jobs)
    }
    fn fallback_static(&mut self, err: GenerateError) -> Result<Vec<StaticRow>, GenerateError> {
        self.inner.fallback_static(err)
    }
    fn ctx_max_seq(&self) -> i32 {
        1
    }
    fn coalesce_limits(&self) -> CoalesceLimits {
        CoalesceLimits::UNBOUNDED
    }
}

impl ContExec for OverflowSpy {
    fn model_id(&self) -> i32 {
        0
    }
    fn seq_cap(&self) -> i32 {
        0
    }
    fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
        vec![77]
    }
    fn encode_text(&self, _text: &str) -> Vec<i32> {
        vec![77]
    }
    fn generate(
        &mut self,
        parsed: &crate::parse::ParsedRequest,
        job_id: &str,
        created: i64,
        cors: bool,
        default_tokens: i32,
        t_arrive: Instant,
        bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        store: Option<&mut ds4_kv::Store>,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        self.inner.generate(
            parsed,
            job_id,
            created,
            cors,
            default_tokens,
            t_arrive,
            bank_hold_retry,
            store,
            out,
        )
    }
    fn as_static(&mut self) -> Option<&mut dyn StaticExec> {
        Some(self)
    }
}
