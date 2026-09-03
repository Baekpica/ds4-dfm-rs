use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::super::*;
use super::run_owner_maybe_roll;
use crate::generate::{GenerateError, GenerateOutcome, ScriptedDecode};
use crate::parse::{parse_request, ParseEnv};
use crate::route::{ThinkMode, WireSurface, LANE_CONTINUOUS};
use crate::serve_cont::{ContExec, ContStepper};
use crate::serve_cont_prefill::{owner_tick_call_count, reset_owner_tick_call_count};
use crate::stream::{ReqTimings, TAPE_PLAIN};
use ds4_kv::Store as KvStore;

struct RollSpy {
    generate_calls: usize,
    max_seq: i32,
}

impl Default for RollSpy {
    fn default() -> Self {
        Self {
            generate_calls: 0,
            max_seq: 2,
        }
    }
}

impl ContExec for RollSpy {
    fn model_id(&self) -> i32 {
        0
    }
    fn seq_cap(&self) -> i32 {
        8192
    }
    fn max_seq(&self) -> i32 {
        self.max_seq
    }
    fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
        vec![1, 2, 3]
    }
    fn encode_text(&self, _text: &str) -> Vec<i32> {
        vec![1, 2, 3]
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
        self.generate_calls += 1;
        out.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}")
            .map_err(|_| GenerateError::Io)?;
        Ok(GenerateOutcome {
            generation: 1,
            frontier: 1,
            finish: "stop".into(),
            ..GenerateOutcome::default()
        })
    }
}

#[test]
fn one_bank_leaves_the_second_request_queued_for_serial_execution() {
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = cont_chat_job(&inner, "a");
    let (job_b, drain_b) = cont_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    tx.send(job_b).unwrap();
    let mut spy = RollSpy {
        max_seq: 1,
        ..RollSpy::default()
    };
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);

    let leftover = run_owner_maybe_roll(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);

    assert!(leftover.is_none());
    assert_eq!(spy.generate_calls, 1);
    let job_b = rx.try_recv().expect("second request must remain queued");
    let leftover = run_owner_maybe_roll(&cfg, &inner, &mut engine, &mut spy, job_b, &rx);
    assert!(leftover.is_none());
    assert_eq!(spy.generate_calls, 2);
    drop(tx);
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
}

fn cont_chat_job(inner: &Arc<Mutex<ServerInner>>, tag: &str) -> (OwnerJob, JobDrain) {
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
fn owner_roll_pair_invokes_tick_roll_prefill() {
    reset_owner_tick_call_count();
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = cont_chat_job(&inner, "a");
    let (job_b, drain_b) = cont_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    tx.send(job_b).unwrap();
    drop(tx);
    let mut spy = RollSpy::default();
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);

    let leftover = run_owner_maybe_roll(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);

    assert!(leftover.is_none());
    assert_eq!(spy.generate_calls, 2);
    assert!(
        owner_tick_call_count() >= 1,
        "production rolling owner must call tick_roll_prefill"
    );
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
}

#[test]
fn late_roll_sibling_joins_when_coalesce_wait_is_set() {
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = cont_chat_job(&inner, "a");
    let (job_b, drain_b) = cont_chat_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(30));
        tx.send(job_b).unwrap();
    });
    let mut spy = RollSpy::default();
    let mut engine = ScriptedDecode::from_pieces(&[b"serial-fallback"]);
    std::env::set_var("DS4_SERVER_COALESCE_WAIT_MS", "100");
    let leftover = run_owner_maybe_roll(&cfg, &inner, &mut engine, &mut spy, job_a, &rx);
    std::env::remove_var("DS4_SERVER_COALESCE_WAIT_MS");

    assert!(leftover.is_none());
    assert_eq!(spy.generate_calls, 2);
    let _ = drain_a.done.recv();
    let _ = drain_b.done.recv();
}

struct StepperCont;

impl ContExec for StepperCont {
    fn model_id(&self) -> i32 {
        0
    }
    fn seq_cap(&self) -> i32 {
        8192
    }
    fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
        vec![1]
    }
    fn encode_text(&self, _text: &str) -> Vec<i32> {
        vec![1]
    }
    fn generate(
        &mut self,
        parsed: &crate::parse::ParsedRequest,
        job_id: &str,
        created: i64,
        cors: bool,
        default_tokens: i32,
        _t_arrive: Instant,
        _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut KvStore>,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        let (mut stepper, head) = ContStepper::new(
            parsed,
            self.model_id(),
            job_id,
            created,
            cors,
            default_tokens,
            b"prompt".to_vec(),
            1,
            self.seq_cap(),
        );
        out.write_all(&head).map_err(|_| GenerateError::Io)?;
        for piece in TAPE_PLAIN {
            let step = stepper.feed(piece.as_bytes());
            out.write_all(&step.bytes).map_err(|_| GenerateError::Io)?;
            if step.done {
                break;
            }
        }
        let (tail, outcome) = stepper.finalize(true, 0, 1, ReqTimings::default(), cors);
        out.write_all(&tail).map_err(|_| GenerateError::Io)?;
        Ok(outcome)
    }
}

#[derive(Default)]
struct PublishingSpy {
    calls: usize,
}

impl ContExec for PublishingSpy {
    fn model_id(&self) -> i32 {
        0
    }
    fn seq_cap(&self) -> i32 {
        8192
    }
    fn max_seq(&self) -> i32 {
        4
    }
    fn encode_chat(&self, _rendered: &[u8]) -> Vec<i32> {
        vec![1]
    }
    fn encode_text(&self, _text: &str) -> Vec<i32> {
        vec![1]
    }
    fn generate(
        &mut self,
        _parsed: &crate::parse::ParsedRequest,
        _job_id: &str,
        _created: i64,
        cors: bool,
        _default_tokens: i32,
        _t_arrive: Instant,
        _bank_hold_retry: &mut dyn FnMut(i32, Option<(u64, i32)>) -> Option<i32>,
        _store: Option<&mut KvStore>,
        out: &mut dyn Write,
    ) -> Result<GenerateOutcome, GenerateError> {
        self.calls += 1;
        out.write_all(&http_response_bytes(
            200,
            Some("application/json"),
            None,
            cors,
            "{}",
        ))
        .map_err(|_| GenerateError::Io)?;
        Ok(GenerateOutcome {
            tool_ids: vec![format!("toolu_pair{}", self.calls)],
            bank: Some((self.calls - 1) as i32),
            generation: 10 + self.calls as u64,
            frontier: 100 + self.calls as i32,
            finish: "tool_calls".into(),
        })
    }
}

fn anthropic_tool_job(inner: &Arc<Mutex<ServerInner>>, tag: &str) -> (OwnerJob, JobDrain) {
    let env = parse_env();
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":"{tag}"}}],"max_tokens":8,"stream":true,"thinking":{{"type":"disabled"}},"tools":[{{"name":"bash","input_schema":{{"type":"object"}}}}]}}"#
    );
    let parsed = parse_request(WireSurface::Anthropic, &env, &body).unwrap();
    let prepared = PreparedJob {
        parsed,
        surface: WireSurface::Anthropic,
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
fn paired_tool_turns_publish_each_actual_bank() {
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 8,
        ..ServerConfig::default()
    };
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let (job_a, drain_a) = anthropic_tool_job(&inner, "a");
    let (job_b, drain_b) = anthropic_tool_job(&inner, "b");
    let (tx, rx) = mpsc::channel();
    tx.send(job_b).unwrap();
    drop(tx);
    let mut spy = PublishingSpy::default();
    let mut engine = ScriptedDecode::from_pieces(&[b"unused"]);

    assert!(run_owner_maybe_roll(&cfg, &inner, &mut engine, &mut spy, job_a, &rx).is_none());
    drain_a
        .done
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("first pair job settled");
    drain_b
        .done
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("second pair job settled");

    let mut g = lock_inner(&inner);
    assert_eq!(
        g.creg
            .bank_claim(Api::Anthropic, &["toolu_pair1".into()], monotonic_now()),
        Some((0, 11, 101))
    );
    assert_eq!(
        g.creg
            .bank_claim(Api::Anthropic, &["toolu_pair2".into()], monotonic_now()),
        Some((1, 12, 102))
    );
}

fn parse_env() -> ParseEnv {
    ParseEnv {
        default_model: "ds4".into(),
        default_tokens: 16,
        default_effort: ThinkMode::None,
        default_temp: 0.0,
        live_ids: Vec::new(),
    }
}

fn drive_cont_generate(
    surface: WireSurface,
    body: &str,
) -> (Result<GenerateOutcome, GenerateError>, String) {
    let parsed = parse_request(surface, &parse_env(), body).unwrap();
    let mut cont = StepperCont;
    let mut out = Vec::new();
    let mut hold = |_bank, _live| None;
    let result = cont.generate(
        &parsed,
        "job-cont-stream",
        1,
        false,
        16,
        Instant::now(),
        &mut hold,
        None,
        &mut out,
    );
    (result, String::from_utf8(out).unwrap())
}

fn drive_streaming_http(path: &str, body: &str) -> (String, ServerInner) {
    let cfg = ServerConfig {
        have_engine: true,
        default_tokens: 16,
        ..ServerConfig::default()
    };
    let inner = Mutex::new(ServerInner::from_cfg(&cfg));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    write!(
        client,
        "POST {path} HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    client.shutdown(std::net::Shutdown::Write).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    let mut engine = ScriptedDecode::from_pieces(&[]);
    let mut cont = StepperCont;
    handle_client_inner(
        &cfg,
        &inner,
        &mut server,
        Some(&mut engine),
        Some(&mut cont),
    );
    drop(server);
    let mut response = Vec::new();
    client.read_to_end(&mut response).unwrap();
    (
        String::from_utf8(response).unwrap(),
        inner.into_inner().unwrap(),
    )
}

#[test]
fn streaming_anthropic_cont_exec_does_not_return_unsupported() {
    // Given: a streaming Anthropic no-tools greedy request
    // When: ContExec::generate runs the continuous stepper
    // Then: it must not return Unsupported (that is run_engine's serial fallback)
    let (result, wire) = drive_cont_generate(
        WireSurface::Anthropic,
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true,"temperature":0,"thinking":{"type":"disabled"}}"#,
    );
    assert!(
        !matches!(result, Err(GenerateError::Unsupported(_))),
        "{result:?}"
    );
    let outcome = result.expect("continuous generate");
    assert_eq!(outcome.finish, "stop");
    assert!(wire.contains("event: message_start"), "{wire}");
}

#[test]
fn streaming_responses_cont_exec_does_not_return_unsupported() {
    // Given: a streaming Responses no-tools greedy request
    // When: ContExec::generate runs the continuous stepper
    // Then: it must not return Unsupported (that is run_engine's serial fallback)
    let (result, wire) = drive_cont_generate(
        WireSurface::Responses,
        r#"{"input":"hi","max_output_tokens":8,"stream":true,"temperature":0,"reasoning":{"effort":"none"}}"#,
    );
    assert!(
        !matches!(result, Err(GenerateError::Unsupported(_))),
        "{result:?}"
    );
    let outcome = result.expect("continuous generate");
    assert_eq!(outcome.finish, "stop");
    assert!(wire.contains("\"type\":\"response.created\""), "{wire}");
}

#[test]
fn streaming_anthropic_settles_continuous_with_message_start() {
    let (response, inner) = drive_streaming_http(
        "/v1/messages",
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true,"temperature":0,"thinking":{"type":"disabled"}}"#,
    );
    assert!(response.starts_with("HTTP/1.1 200 "), "{response}");
    assert!(response.contains("event: message_start"), "{response}");
    assert_eq!(
        inner.metrics.route_requests[WireSurface::Anthropic as usize][LANE_CONTINUOUS as usize],
        1
    );
    assert_eq!(inner.runtime.requests_serial, 0);
}

#[test]
fn streaming_responses_settles_continuous_with_response_created() {
    let (response, inner) = drive_streaming_http(
        "/v1/responses",
        r#"{"input":"hi","max_output_tokens":8,"stream":true,"temperature":0,"reasoning":{"effort":"none"}}"#,
    );
    assert!(response.starts_with("HTTP/1.1 200 "), "{response}");
    assert!(
        response.contains("\"type\":\"response.created\""),
        "{response}"
    );
    assert_eq!(
        inner.metrics.route_requests[WireSurface::Responses as usize][LANE_CONTINUOUS as usize],
        1
    );
    assert_eq!(inner.runtime.requests_serial, 0);
}
