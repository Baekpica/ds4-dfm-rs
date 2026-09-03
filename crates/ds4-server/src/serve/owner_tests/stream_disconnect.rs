//! Mid-stream disconnect / backpressure fixtures for Anthropic and Responses.
//! Event names stay the C automata; client drop must not panic.

use super::{
    admit_test_job, drain_job, job_sink, lock_inner, owner_job_with_probe, run_owner_job, test_cfg,
    DisconnectDecode, OwnerJob, PreparedJob, Settle, JOB_SINK_CAP_BYTES,
};
use crate::admit::SHED_SLOW_READER;
use crate::generate::{generate_and_write, GenerateError, ScriptedDecode};
use crate::parse::{parse_request, ParseEnv, ParsedRequest};
use crate::route::{ThinkMode, WireSurface};
use crate::stream::TAPE_PLAIN;
use crate::ServerInner;
use std::io::{self, Write};
use std::net::{TcpListener, TcpStream};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const ANTHROPIC_STREAM: &str = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true,"thinking":{"type":"disabled"}}"#;
const RESPONSES_STREAM: &str = r#"{"input":"hi","max_output_tokens":8,"stream":true}"#;

struct FailOnWrite {
    fail_at: usize,
    writes: usize,
    kind: io::ErrorKind,
    captured: Vec<u8>,
}

impl Write for FailOnWrite {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        if self.writes >= self.fail_at {
            return Err(io::Error::new(self.kind, "client gone"));
        }
        self.captured.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.writes >= self.fail_at {
            Err(io::Error::new(self.kind, "client gone"))
        } else {
            Ok(())
        }
    }
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

fn streamed(surface: WireSurface, body: &str) -> ParsedRequest {
    let mut parsed = parse_request(surface, &parse_env(), body).unwrap();
    parsed.think_mode = ThinkMode::None;
    parsed.temperature = 0.0;
    parsed
}

fn tape_engine() -> ScriptedDecode {
    ScriptedDecode::from_pieces(&TAPE_PLAIN.iter().map(|s| s.as_bytes()).collect::<Vec<_>>())
}

fn generate_mid_stream_io(
    surface: WireSurface,
    body: &str,
    job_id: &str,
    kind: io::ErrorKind,
    start_marker: &str,
    stop_marker: &str,
) {
    let parsed = streamed(surface, body);
    let mut engine = tape_engine();
    let mut out = FailOnWrite {
        // Headers are sent before prefill; protocol start is the second write.
        fail_at: 3,
        writes: 0,
        kind,
        captured: Vec::new(),
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        generate_and_write(&mut engine, &parsed, job_id, 1, false, 16, &mut out)
    }));
    let err = result
        .expect("client drop must not panic")
        .expect_err("mid-stream write fail must abort");
    assert!(matches!(err, GenerateError::Io), "{err}");
    let prefix = String::from_utf8_lossy(&out.captured);
    assert!(prefix.contains(start_marker), "{prefix}");
    assert!(!prefix.contains(stop_marker), "{prefix}");
}

fn streamed_job(surface: WireSurface, body: &str, body_bytes: u64) -> PreparedJob {
    PreparedJob {
        parsed: streamed(surface, body),
        cont_prompt: None,
        surface,
        body_bytes,
        arrived_at: Instant::now(),
    }
}

fn assert_slow_sink_cancels(surface: WireSurface, body: &str) {
    let cfg = test_cfg();
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let mut prepared = streamed_job(surface, body, 31);
    let lease = admit_test_job(&inner, &mut prepared);
    let (mut sink, state) = job_sink(Arc::clone(&inner));
    sink.write_all(&vec![0; JOB_SINK_CAP_BYTES as usize])
        .unwrap();
    let (done_tx, done_rx) = mpsc::channel();
    let job = OwnerJob {
        lease,
        prepared,
        sink,
        done: done_tx,
    };
    let mut engine = tape_engine();
    run_owner_job(&cfg, &inner, &mut engine, None, job);
    assert!(state.slow());
    let lease = done_rx.recv().unwrap();
    assert_eq!(lease.settlement.outcome, Settle::Canceled);
    drop(lease);
    let g = lock_inner(&inner);
    assert_eq!(g.runtime.requests_canceled, 1);
    assert_eq!(g.runtime.requests_failed, 0);
    assert_eq!(g.metrics.shed[SHED_SLOW_READER as usize], 1);
}

fn assert_client_drop_cancels(surface: WireSurface, body: &str) {
    let cfg = test_cfg();
    let inner = Arc::new(Mutex::new(ServerInner::from_cfg(&cfg)));
    let mut prepared = streamed_job(surface, body, 23);
    prepared.parsed.max_tokens = 5;
    prepared.parsed.max_tokens_set = true;
    let lease = admit_test_job(&inner, &mut prepared);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, _) = listener.accept().unwrap();
    server.set_nonblocking(true).unwrap();
    let probe = server.try_clone().unwrap();
    let (job, drain) = owner_job_with_probe(prepared, lease, Some(probe));
    let state = Arc::clone(&drain.state);
    let drain_thread = thread::spawn(move || drain_job(server, drain));
    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let disconnect = thread::spawn(move || {
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        client.shutdown(std::net::Shutdown::Both).unwrap();
        drop(client);
        resume_tx.send(()).unwrap();
    });
    let mut engine = DisconnectDecode {
        started: Some(started_tx),
        resume: resume_rx,
        samples: 0,
        evals: 0,
        pos: 0,
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        run_owner_job(&cfg, &inner, &mut engine, None, job);
    }));
    assert!(result.is_ok(), "client drop must not panic");
    disconnect.join().unwrap();
    drain_thread.join().unwrap();
    assert!(state.gone(), "decode probe must observe the TCP disconnect");
    assert!(
        engine.evals <= 1,
        "decode continued for {} tokens",
        engine.evals
    );
    let g = lock_inner(&inner);
    assert_eq!(g.runtime.requests_canceled, 1);
    assert_eq!(g.runtime.requests_failed, 0);
    assert_eq!(g.runtime.requests_inflight, 0);
}

#[test]
fn anthropic_stream_broken_pipe_mid_delta_is_io_without_panic() {
    generate_mid_stream_io(
        WireSurface::Anthropic,
        ANTHROPIC_STREAM,
        "msg-drop",
        io::ErrorKind::BrokenPipe,
        "event: message_start",
        "event: message_stop",
    );
}

#[test]
fn responses_stream_broken_pipe_mid_delta_is_io_without_panic() {
    generate_mid_stream_io(
        WireSurface::Responses,
        RESPONSES_STREAM,
        "resp-drop",
        io::ErrorKind::BrokenPipe,
        "\"type\":\"response.created\"",
        "\"type\":\"response.completed\"",
    );
}

#[test]
fn anthropic_stream_would_block_mid_delta_is_io_without_panic() {
    generate_mid_stream_io(
        WireSurface::Anthropic,
        ANTHROPIC_STREAM,
        "msg-slow",
        io::ErrorKind::WouldBlock,
        "event: message_start",
        "event: message_stop",
    );
}

#[test]
fn responses_stream_would_block_mid_delta_is_io_without_panic() {
    generate_mid_stream_io(
        WireSurface::Responses,
        RESPONSES_STREAM,
        "resp-slow",
        io::ErrorKind::WouldBlock,
        "\"type\":\"response.created\"",
        "\"type\":\"response.completed\"",
    );
}

#[test]
fn anthropic_stream_slow_sink_cancels_once() {
    assert_slow_sink_cancels(WireSurface::Anthropic, ANTHROPIC_STREAM);
}

#[test]
fn responses_stream_slow_sink_cancels_once() {
    assert_slow_sink_cancels(WireSurface::Responses, RESPONSES_STREAM);
}

#[test]
fn anthropic_stream_client_drop_mid_decode_cancels_without_panic() {
    assert_client_drop_cancels(WireSurface::Anthropic, ANTHROPIC_STREAM);
}

#[test]
fn responses_stream_client_drop_mid_decode_cancels_without_panic() {
    assert_client_drop_cancels(WireSurface::Responses, RESPONSES_STREAM);
}
