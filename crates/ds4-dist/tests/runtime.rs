//! Coordinator/worker runtime: CLI strings, route search, HELLO/WORK/RESULT.

use ds4_dist::{
    build_route_plan, decode_logits_payload, decode_result_body, dispatch_eval, encode_route_blob,
    encode_work_frame, format_telemetry_line, parse_cli, parse_layers, parse_role,
    prefetch_disabled_from, prepare_engine_options, read_frame, reconnect_local, register_worker,
    resolved_layer_end, send_hello, serve_prefetch_local_with, token_hash_update_span,
    token_span_hashes, validate_layers_for_model, validate_options, work_with_ids, write_frame,
    Coordinator, CoordinatorView, EvalOutcome, JobQueue, LocalReconnect, PrefetchJob, ReturnTarget,
    RouteEntry, SliceExec, Telemetry, Work, WorkBody, WorkOutput, WorkRequest, Worker, WorkerInfo,
    ERR_NEXT_CLOSED, MSG_ERROR, MSG_RESULT, RESULT_LOGITS, ROUTE_F_OUTPUT_LOGITS,
    ROUTE_RETURN_UPSTREAM, TOKEN_HASH_INIT, WORK_F_INPUT_HC, WORK_F_OUTPUT_LOGITS,
    WORK_F_RESET_SESSION,
};
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

fn oracle() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("DS4_DIST_C_ORACLE") {
        return std::path::PathBuf::from(p);
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/dist_c_oracle")
}

fn require_oracle() -> std::path::PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/dist_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_out(args: &[&str]) -> String {
    let out = std::process::Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run dist_c_oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn rust_cli(args: &[&str]) -> Result<(), String> {
    let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    let (opt, unmatched) = parse_cli(&owned)?;
    if let Some(u) = unmatched.first() {
        return Err(format!("unmatched {u}"));
    }
    validate_options(&opt)
}

fn rust_cli_layers(n_layers: u32, args: &[&str]) -> Result<(), String> {
    let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    let (opt, unmatched) = parse_cli(&owned)?;
    if let Some(u) = unmatched.first() {
        return Err(format!("unmatched {u}"));
    }
    validate_options(&opt)?;
    validate_layers_for_model(&opt, n_layers)
}

fn assert_cli(args: &[&str]) {
    let mut cargs = vec!["cli"];
    cargs.extend_from_slice(args);
    let c = c_out(&cargs);
    let r = rust_cli(args);
    if c == "OK" {
        r.expect("rust rejected a CLI C accepts");
    } else {
        let msg = c.strip_prefix("ERROR:").unwrap_or(&c);
        let err = r.expect_err("rust accepted a CLI C rejects");
        assert_eq!(err, msg, "args={args:?}");
    }
}

#[derive(Clone)]
struct MockExec {
    model_id: u32,
    n_layers: u32,
    vocab: u32,
    ctx_size: u32,
    hidden_values: u64,
    has_output: bool,
    layer_start: u32,
    layer_end: u32,
    hidden: Vec<f32>,
    logits: Vec<f32>,
}

impl SliceExec for MockExec {
    fn model_id(&self) -> u32 {
        self.model_id
    }
    fn n_layers(&self) -> u32 {
        self.n_layers
    }
    fn vocab(&self) -> u32 {
        self.vocab
    }
    fn ctx_size(&self) -> u32 {
        self.ctx_size
    }
    fn hidden_values(&self) -> u64 {
        self.hidden_values
    }
    fn has_output(&self) -> bool {
        self.has_output
    }
    fn layer_start(&self) -> u32 {
        self.layer_start
    }
    fn layer_end(&self) -> u32 {
        self.layer_end
    }
    fn eval(&mut self, req: &WorkRequest) -> Result<WorkOutput, String> {
        Ok(WorkOutput {
            hidden: if req.produce_hidden {
                Some(self.hidden.clone())
            } else {
                None
            },
            logits: if req.produce_logits {
                Some(self.logits.clone())
            } else {
                None
            },
        })
    }
}

fn tune(s: &TcpStream) {
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
}

#[test]
fn telemetry_line_matches_c_format() {
    let tel = Telemetry {
        layer_start: 2,
        layer_end: 3,
        route_index: 1,
        pos0: 4,
        n_tokens: 2,
        eval_usec: 1500,
        downstream_wait_usec: 250,
        forward_send_usec: 100,
        input_bytes: 2 * 1024 * 1024,
        output_bytes: 512 * 1024,
    };
    assert_eq!(
        format_telemetry_line(9, 0, &tel),
        "ds4: distributed telemetry: request=9 hop=0 layers=2:3 route=1 pos=4 tokens=2 eval=1.500ms downstream_wait=0.250ms forward_send=0.100ms input=2.00MiB output=0.50MiB\n"
    );
}

#[test]
fn layers_and_role_match_c() {
    assert_eq!(c_out(&["layers", "0:1"]), "start=0 end=1 has_output=0");
    assert_eq!(
        c_out(&["layers", "21:output"]),
        "start=21 end=4294967295 has_output=1"
    );
    let l = parse_layers("21:output").unwrap();
    assert_eq!(l.start, 21);
    assert!(l.has_output);
    assert_eq!(resolved_layer_end(&l, 53), 52);
    assert_eq!(c_out(&["layers", "foo"]), "ERROR:expected A:B or A:output");
    assert_eq!(parse_layers("foo").unwrap_err(), "expected A:B or A:output");
    assert_eq!(
        c_out(&["layers", "x:1"]),
        "ERROR:invalid start layer in x:1"
    );
    assert_eq!(
        parse_layers("x:1").unwrap_err(),
        "invalid start layer in x:1"
    );
    assert_eq!(
        c_out(&["layers", "3:1"]),
        "ERROR:layer range end precedes start in 3:1"
    );
    assert_eq!(c_out(&["role", "coordinator"]), "coordinator");
    assert_eq!(parse_role("coordinator"), Some(ds4_dist::Role::Coordinator));
    assert!(c_out(&["role", "boss"]).starts_with("ERROR:invalid distributed role: boss"));
    assert!(parse_role("boss").is_none());
}

#[test]
fn cli_validate_matches_c() {
    assert_cli(&[]);
    assert_cli(&["--listen", "127.0.0.1", "7000"]);
    assert_cli(&["--role", "coordinator"]);
    assert_cli(&[
        "--role",
        "coordinator",
        "--layers",
        "0:1",
        "--listen",
        "127.0.0.1",
        "7000",
    ]);
    assert_cli(&[
        "--role",
        "coordinator",
        "--layers",
        "0:1",
        "--listen",
        "127.0.0.1",
        "7000",
        "--coordinator",
        "10.0.0.1",
        "9",
    ]);
    assert_cli(&["--role", "worker", "--layers", "2:output"]);
    assert_cli(&[
        "--role",
        "worker",
        "--layers",
        "2:output",
        "--coordinator",
        "127.0.0.1",
        "7000",
    ]);
    assert_cli(&[
        "--role",
        "worker",
        "--layers",
        "2:output",
        "--coordinator",
        "127.0.0.1",
        "7000",
        "--dist-prefill-chunk",
        "128",
    ]);
    assert_cli(&["--role", "nope"]);
    assert_cli(&["--layers", "0:1:2"]);
    assert_cli(&["--dist-prefill-window", "99"]);
    assert_cli(&[
        "--role",
        "coordinator",
        "--layers",
        "0:1",
        "--listen",
        "127.0.0.1",
        "0",
    ]);
}

#[test]
fn layers_for_model_match_c() {
    let args = [
        "--role",
        "coordinator",
        "--layers",
        "2:3",
        "--listen",
        "127.0.0.1",
        "7000",
    ];
    let c = c_out(&[
        "layers-for-model",
        "4",
        "--role",
        "coordinator",
        "--layers",
        "2:3",
        "--listen",
        "127.0.0.1",
        "7000",
    ]);
    let r = rust_cli_layers(4, &args);
    assert_eq!(c, "ERROR:coordinator layer range must start at layer 0");
    assert_eq!(
        r.unwrap_err(),
        "coordinator layer range must start at layer 0"
    );

    let past = rust_cli_layers(
        4,
        &[
            "--role",
            "worker",
            "--layers",
            "9:9",
            "--coordinator",
            "127.0.0.1",
            "1",
        ],
    );
    assert_eq!(
        past.unwrap_err(),
        "layer range starts past final model layer 3"
    );
}

#[test]
fn replay_check_requires_coordinator() {
    let owned = vec![
        "--role".into(),
        "worker".into(),
        "--layers".into(),
        "2:output".into(),
        "--coordinator".into(),
        "127.0.0.1".into(),
        "7000".into(),
        "--dist-replay-check".into(),
    ];
    let (opt, _) = parse_cli(&owned).unwrap();
    validate_options(&opt).unwrap();
    assert_eq!(
        prepare_engine_options(&opt).unwrap_err(),
        "--dist-replay-check requires --role coordinator"
    );
}

fn worker(start: u32, end: u32, output: bool) -> WorkerInfo {
    WorkerInfo {
        peer_host: format!("10.0.0.{end}"),
        listen_port: 7000 + end,
        model_id: 3,
        quant_bits: 2,
        layer_start: start,
        layer_end: end,
        has_output: output,
        has_hidden: true,
    }
}

#[test]
fn route_plan_local_complete_is_empty() {
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 3,
        local_has_output: true,
        local_can_output_head: false,
    };
    let plan = build_route_plan(&view, &[]).unwrap();
    assert!(plan.entries.is_empty());
    assert!(plan.blob.is_empty());
}

#[test]
fn route_plan_requires_layer_zero() {
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 1,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    assert_eq!(
        build_route_plan(&view, &[]).unwrap_err(),
        "coordinator route does not start at layer 0"
    );
}

#[test]
fn route_plan_missing_layer_message() {
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    assert_eq!(
        build_route_plan(&view, &[]).unwrap_err(),
        "distributed route incomplete: missing layer 2"
    );
}

#[test]
fn route_plan_prefers_output_then_longer_end() {
    let view = CoordinatorView {
        n_layers: 6,
        local_start: 0,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    let workers = vec![worker(2, 3, false), worker(2, 5, true), worker(2, 4, false)];
    let plan = build_route_plan(&view, &workers).unwrap();
    assert_eq!(plan.entries.len(), 1);
    assert_eq!(plan.entries[0].layer_end, 5);
    assert_ne!(plan.entries[0].flags, 0);
}

#[test]
fn register_worker_replaces_stale_and_prepends() {
    let mut list = vec![worker(2, 5, true)];
    list[0].peer_host = "10.0.0.9".into();
    let neu = WorkerInfo {
        peer_host: "10.0.0.9".into(),
        listen_port: 7101,
        model_id: 3,
        quant_bits: 4,
        layer_start: 2,
        layer_end: 5,
        has_output: true,
        has_hidden: true,
    };
    register_worker(&mut list, neu.clone());
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].listen_port, 7101);
    let other = worker(2, 3, false);
    register_worker(&mut list, other);
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].layer_end, 3);
}

#[test]
fn single_hop_hello_work_logits() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let logits = vec![0.1f32, 0.2, 0.7, 0.0];
    let hidden = vec![1.0f32, 2.0, 3.0, 4.0];
    let exec = MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: true,
        layer_start: 2,
        layer_end: 3,
        hidden: hidden.clone(),
        logits: logits.clone(),
    };
    let worker_thread = thread::spawn(move || {
        let mut stream = TcpStream::connect(addr).unwrap();
        tune(&stream);
        let mut worker = Worker::new(exec);
        let (hello, name) = worker.hello(7000, 128, "mock");
        send_hello(&mut stream, &hello, &name).unwrap();
        worker.serve(&mut stream).unwrap();
    });

    let (mut stream, _) = listener.accept().unwrap();
    tune(&stream);
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    let mut coord = Coordinator::new(view, 3, 32);
    coord.accept_hello(&mut stream, "127.0.0.1").unwrap();
    let tokens = [1i32, 2];
    let out = dispatch_eval(
        &coord,
        &mut stream,
        &tokens,
        0,
        9,
        11,
        true,
        TOKEN_HASH_INIT,
        &hidden,
    )
    .unwrap();
    match out {
        EvalOutcome::Logits(v) => assert_eq!(v, logits),
        other => panic!("expected logits, got {other:?}"),
    }
    drop(stream);
    worker_thread.join().unwrap();
}

#[test]
fn two_hop_blocking_forward() {
    let data = TcpListener::bind("127.0.0.1:0").unwrap();
    let data_port = data.local_addr().unwrap().port() as u32;
    let coord_l = TcpListener::bind("127.0.0.1:0").unwrap();
    let coord_addr = coord_l.local_addr().unwrap();
    let (ready_tx, ready_rx) = mpsc::channel::<()>();

    let tail = MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: true,
        layer_start: 2,
        layer_end: 3,
        hidden: vec![9.0, 8.0],
        logits: vec![0.0, 1.0, 0.0, 0.0],
    };
    let mid = MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: false,
        layer_start: 1,
        layer_end: 1,
        hidden: vec![0.5, 1.5],
        logits: vec![],
    };

    let tail_thread = thread::spawn(move || {
        let mut hello_s = TcpStream::connect(coord_addr).unwrap();
        tune(&hello_s);
        let mut worker = Worker::new(tail);
        let (hello, name) = worker.hello(data_port, 128, "tail");
        send_hello(&mut hello_s, &hello, &name).unwrap();
        ready_tx.send(()).unwrap();
        let (mut work_s, _) = data.accept().unwrap();
        tune(&work_s);
        worker.serve(&mut work_s).unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();

    let mid_thread = thread::spawn(move || {
        let mut stream = TcpStream::connect(coord_addr).unwrap();
        tune(&stream);
        let mut worker = Worker::new(mid);
        let (hello, name) = worker.hello(7100, 128, "mid");
        send_hello(&mut stream, &hello, &name).unwrap();
        worker.serve(&mut stream).unwrap();
    });

    let mut first = None;
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 0,
        local_has_output: false,
        local_can_output_head: false,
    };
    let mut coord = Coordinator::new(view, 3, 32);
    for _ in 0..2 {
        let (mut s, _) = coord_l.accept().unwrap();
        tune(&s);
        let (hello, _) = coord.accept_hello(&mut s, "127.0.0.1").unwrap();
        if hello.layer_start == 1 {
            first = Some(s);
        }
    }
    let mut stream = first.expect("mid worker stream");
    let tokens = [3i32];
    let hc = vec![0.25f32, 0.75];
    let out = dispatch_eval(
        &coord,
        &mut stream,
        &tokens,
        0,
        1,
        2,
        true,
        TOKEN_HASH_INIT,
        &hc,
    )
    .unwrap();
    match out {
        EvalOutcome::Logits(v) => assert_eq!(v, vec![0.0, 1.0, 0.0, 0.0]),
        other => panic!("expected logits, got {other:?}"),
    }
    drop(stream);
    mid_thread.join().unwrap();
    tail_thread.join().unwrap();
}

#[test]
fn worker_rejects_model_mismatch_with_c_string() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let exec = MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: true,
        layer_start: 2,
        layer_end: 3,
        hidden: vec![0.0; 4],
        logits: vec![1.0; 4],
    };
    let worker_thread = thread::spawn(move || {
        let mut stream = TcpStream::connect(addr).unwrap();
        tune(&stream);
        let mut worker = Worker::new(exec);
        let (hello, name) = worker.hello(7000, 128, "mock");
        send_hello(&mut stream, &hello, &name).unwrap();
        worker.serve(&mut stream).unwrap();
    });
    let (mut stream, _) = listener.accept().unwrap();
    tune(&stream);
    let view = CoordinatorView {
        n_layers: 4,
        local_start: 0,
        local_end: 1,
        local_has_output: false,
        local_can_output_head: false,
    };
    let mut coord = Coordinator::new(view, 99, 32);
    coord.accept_hello(&mut stream, "127.0.0.1").unwrap();
    let err = dispatch_eval(
        &coord,
        &mut stream,
        &[1],
        0,
        1,
        1,
        true,
        TOKEN_HASH_INIT,
        &[0.0, 0.0],
    )
    .unwrap_err();
    assert_eq!(err, "model id mismatch: work=99 worker=3");
    drop(stream);
    worker_thread.join().unwrap();
}

#[test]
fn prefix_hash_continues_committed_timeline() {
    let committed = [7i32, 8];
    let span = [9i32];
    let (prefix, result) = token_span_hashes(&committed, &span);
    assert_eq!(prefix, ds4_dist::token_hash_prefix(&committed));
    assert_eq!(result, ds4_dist::token_hash_update_span(prefix, &span));
    assert_ne!(prefix, TOKEN_HASH_INIT);
}

#[test]
fn usage_text_mentions_roles() {
    assert!(ds4_dist::USAGE.contains("--role ROLE"));
    assert!(ds4_dist::USAGE.contains("21:output"));
}

#[test]
fn open_data_listener_assigns_ephemeral_port() {
    let (listener, port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
    assert_ne!(port, 0);
    assert_eq!(listener.local_addr().unwrap().port(), port);
    let _stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
}

#[test]
fn open_data_listener_accept_data_client_returns_client() {
    let (listener, port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
    let connector = thread::spawn(move || TcpStream::connect(("127.0.0.1", port)).unwrap());
    let client = ds4_dist::accept_data_client(&listener).unwrap();
    let _peer = connector.join().unwrap();
    assert!(client.nodelay().unwrap());
}

#[test]
fn open_data_listener_accept_data_client_errors_after_close() {
    use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

    let (listener, _port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
    let raw = listener.into_raw_fd();
    // SAFETY: `raw` is exclusively owned; OwnedFd closes it.
    drop(unsafe { OwnedFd::from_raw_fd(raw) });
    // SAFETY: `raw` is a closed fd. Reconstruct only so accept observes the
    // error, then forget so Drop does not close twice.
    let listener = unsafe { TcpListener::from_raw_fd(raw) };
    let err = ds4_dist::accept_data_client(&listener);
    std::mem::forget(listener);
    assert!(err.is_err());
}

fn hop_logits_exec() -> (MockExec, Vec<f32>) {
    let logits = vec![0.1f32, 0.2, 0.7, 0.0];
    (
        MockExec {
            model_id: 3,
            n_layers: 4,
            vocab: 8,
            ctx_size: 128,
            hidden_values: 2,
            has_output: true,
            layer_start: 2,
            layer_end: 3,
            hidden: vec![1.0, 2.0, 3.0, 4.0],
            logits: logits.clone(),
        },
        logits,
    )
}

fn hop_work_frame(port: u16) -> Vec<u8> {
    hop_work_frame_with_id(port, 11)
}

fn hop_work_frame_with_id(port: u16, request_id: u64) -> Vec<u8> {
    let tokens = [1i32, 2];
    let prefix = TOKEN_HASH_INIT;
    let result = token_hash_update_span(prefix, &tokens);
    let route_blob = encode_route_blob(
        &[RouteEntry {
            host: "127.0.0.1".into(),
            port: u32::from(port),
            layer_start: 2,
            layer_end: 3,
            flags: ROUTE_F_OUTPUT_LOGITS,
        }],
        &ReturnTarget {
            kind: ROUTE_RETURN_UPSTREAM,
            host: String::new(),
            port: 0,
        },
    )
    .unwrap();
    let work = work_with_ids(
        Work {
            model_id: 3,
            pos0: 0,
            n_tokens: tokens.len() as u32,
            layer_start: 2,
            layer_end: 3,
            flags: WORK_F_INPUT_HC | WORK_F_OUTPUT_LOGITS | WORK_F_RESET_SESSION,
            input_hc_bits: 32,
            route_count: 1,
            route_index: 0,
            ..Work::default()
        },
        9,
        request_id,
        prefix,
        result,
    );
    encode_work_frame(&WorkBody {
        work,
        tokens: tokens.to_vec(),
        input_hc: vec![1.0, 2.0, 3.0, 4.0],
        route_blob,
    })
    .unwrap()
}

#[test]
fn accepted_hop_serve_once_returns_result_for_work() {
    // Given: a data listener and a MockExec worker that owns the final hop
    let (listener, port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
    let frame = hop_work_frame(port);
    let client = thread::spawn(move || {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        tune(&stream);
        stream.write_all(&frame).unwrap();
        read_frame(&mut stream).unwrap()
    });

    // When: accept the hop and serve one WORK
    let mut hop = ds4_dist::accept_data_client(&listener).unwrap();
    tune(&hop);
    let (exec, logits) = hop_logits_exec();
    let mut worker = Worker::new(exec);
    worker.serve_once(&mut hop).unwrap();

    // Then: the client reads a RESULT with the mock logits
    let (typ, reply) = client.join().unwrap();
    assert_eq!(typ, MSG_RESULT);
    let result = decode_result_body(&reply).unwrap();
    assert_eq!(result.hdr.status, 0);
    assert_eq!(result.hdr.result_kind, RESULT_LOGITS);
    assert_eq!(decode_logits_payload(&result.payload).unwrap(), logits);
}

#[test]
fn accepted_hop_serve_once_unknown_frame_yields_error() {
    // Given: a data listener and a MockExec worker
    let (listener, port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
    let client = thread::spawn(move || {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        tune(&stream);
        write_frame(&mut stream, 99, b"not-a-work-frame").unwrap();
        read_frame(&mut stream).unwrap()
    });

    // When: accept the hop and serve one unknown frame
    let mut hop = ds4_dist::accept_data_client(&listener).unwrap();
    tune(&hop);
    let (exec, _) = hop_logits_exec();
    let mut worker = Worker::new(exec);
    let err = worker.serve_once(&mut hop).unwrap_err();

    // Then: the worker rejects the type and the client reads an ERROR frame
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    let (typ, body) = client.join().unwrap();
    assert_eq!(typ, MSG_ERROR);
    assert_eq!(body, b"unsupported distributed worker frame");
}

#[test]
fn reconnect_local_keeps_coordinator_open_while_hop_work_completes() {
    // Given: a coordinator TCP and a worker data listener
    let coord_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let coord_addr = coord_listener.local_addr().unwrap();
    let (data_listener, data_port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
    let frame = hop_work_frame(data_port);
    let (stop_tx, stop_rx) = mpsc::channel();
    let (exec, logits) = hop_logits_exec();

    let worker_thread = thread::spawn(move || {
        let mut worker = Worker::new(exec);
        let (hello, name) = worker.hello(u32::from(data_port), 128, "mock");
        let mut stopped = false;
        reconnect_local(
            &mut worker,
            LocalReconnect {
                connect: || {
                    let stream = TcpStream::connect(coord_addr)?;
                    tune(&stream);
                    Ok(stream)
                },
                hello: &hello,
                model_name: &name,
                sleep: || {},
                should_stop: || {
                    stopped |= stop_rx.try_recv().is_ok();
                    stopped
                },
                listener: Some(&data_listener),
            },
        )
    });

    // When: HELLO stays on the coordinator while a hop WORK is served
    let (mut coord, _) = coord_listener.accept().unwrap();
    tune(&coord);
    let _ = ds4_dist::recv_hello(&mut coord).unwrap();

    let mut hop = TcpStream::connect(("127.0.0.1", data_port)).unwrap();
    tune(&hop);
    hop.write_all(&frame).unwrap();
    let (typ, reply) = read_frame(&mut hop).unwrap();

    // Then: hop RESULT arrives and the coordinator TCP is still open
    assert_eq!(typ, MSG_RESULT);
    let result = decode_result_body(&reply).unwrap();
    assert_eq!(result.hdr.status, 0);
    assert_eq!(result.hdr.result_kind, RESULT_LOGITS);
    assert_eq!(decode_logits_payload(&result.payload).unwrap(), logits);
    assert!(coord.peer_addr().is_ok());
    coord
        .set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    let mut probe = [0u8; 1];
    let idle = coord.read(&mut probe);
    assert!(
        matches!(&idle, Err(e) if e.kind() == std::io::ErrorKind::TimedOut
            || e.kind() == std::io::ErrorKind::WouldBlock),
        "coordinator must stay open (got {idle:?})"
    );

    stop_tx.send(()).unwrap();
    drop(coord);
    worker_thread.join().unwrap().unwrap();
}

fn middle_exec() -> MockExec {
    MockExec {
        model_id: 3,
        n_layers: 4,
        vocab: 8,
        ctx_size: 128,
        hidden_values: 2,
        has_output: false,
        layer_start: 0,
        layer_end: 1,
        hidden: vec![1.0, 2.0, 3.0, 4.0],
        logits: Vec::new(),
    }
}

fn two_hop_work_frame(middle_port: u16, next_port: u16) -> Vec<u8> {
    let tokens = [1i32, 2];
    let prefix = TOKEN_HASH_INIT;
    let result = token_hash_update_span(prefix, &tokens);
    let route_blob = encode_route_blob(
        &[
            RouteEntry {
                host: "127.0.0.1".into(),
                port: u32::from(middle_port),
                layer_start: 0,
                layer_end: 1,
                flags: 0,
            },
            RouteEntry {
                host: "127.0.0.1".into(),
                port: u32::from(next_port),
                layer_start: 2,
                layer_end: 3,
                flags: ROUTE_F_OUTPUT_LOGITS,
            },
        ],
        &ReturnTarget {
            kind: ROUTE_RETURN_UPSTREAM,
            host: String::new(),
            port: 0,
        },
    )
    .unwrap();
    let work = work_with_ids(
        Work {
            model_id: 3,
            pos0: 0,
            n_tokens: tokens.len() as u32,
            layer_start: 0,
            layer_end: 1,
            flags: WORK_F_RESET_SESSION,
            input_hc_bits: 32,
            route_count: 2,
            route_index: 0,
            ..Work::default()
        },
        9,
        11,
        prefix,
        result,
    );
    encode_work_frame(&WorkBody {
        work,
        tokens: tokens.to_vec(),
        input_hc: Vec::new(),
        route_blob,
    })
    .unwrap()
}

fn spawn_reconnect_worker(
    exec: MockExec,
    coord_addr: SocketAddr,
    data_listener: TcpListener,
    data_port: u16,
    stop_rx: mpsc::Receiver<()>,
) -> thread::JoinHandle<std::io::Result<()>> {
    thread::spawn(move || {
        let mut worker = Worker::new(exec);
        let (hello, name) = worker.hello(u32::from(data_port), 128, "mock");
        let mut stopped = false;
        reconnect_local(
            &mut worker,
            LocalReconnect {
                connect: || {
                    let stream = TcpStream::connect(coord_addr)?;
                    tune(&stream);
                    Ok(stream)
                },
                hello: &hello,
                model_name: &name,
                sleep: || {},
                should_stop: || {
                    stopped |= stop_rx.try_recv().is_ok();
                    stopped
                },
                listener: Some(&data_listener),
            },
        )
    })
}

fn accept_hello(listener: &TcpListener) -> TcpStream {
    let (stream, _) = listener.accept().unwrap();
    tune(&stream);
    let mut stream = stream;
    let _ = ds4_dist::recv_hello(&mut stream).unwrap();
    stream
}

fn assert_c_shaped_telemetry(request_id: u64, hop: u32, tel: &Telemetry) {
    let line = format_telemetry_line(request_id, hop, tel);
    assert!(
        line.starts_with("ds4: distributed telemetry: "),
        "C telemetry prefix missing: {line:?}"
    );
    assert!(line.contains(&format!("request={request_id}")));
    assert!(line.contains(&format!("hop={hop}")));
    assert!(line.contains(&format!("layers={}:{}", tel.layer_start, tel.layer_end)));
    assert!(line.contains("eval="));
    assert!(line.contains("downstream_wait="));
    assert!(line.contains("forward_send="));
    assert!(line.contains("input="));
    assert!(line.contains("output="));
    assert!(line.contains("MiB"));
    assert!(line.ends_with('\n'));
}

#[test]
fn reconnect_local_two_hop_result_includes_c_shaped_telemetry() {
    // Given: production reconnect_local mux on a middle worker and a final hop
    let coord_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let coord_addr = coord_listener.local_addr().unwrap();
    let (middle_listener, middle_port) =
        ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
    let (final_listener, final_port) = ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
    let (stop_mid_tx, stop_mid_rx) = mpsc::channel();
    let (stop_fin_tx, stop_fin_rx) = mpsc::channel();
    let (final_exec, logits) = hop_logits_exec();

    let final_thread = spawn_reconnect_worker(
        final_exec,
        coord_addr,
        final_listener,
        final_port,
        stop_fin_rx,
    );
    let middle_thread = spawn_reconnect_worker(
        middle_exec(),
        coord_addr,
        middle_listener,
        middle_port,
        stop_mid_rx,
    );
    let coord_final = accept_hello(&coord_listener);
    let coord_middle = accept_hello(&coord_listener);

    // When: a two-hop WORK is sent on the middle worker data port
    let frame = two_hop_work_frame(middle_port, final_port);
    let mut hop = TcpStream::connect(("127.0.0.1", middle_port)).unwrap();
    tune(&hop);
    hop.write_all(&frame).unwrap();
    let (typ, reply) = read_frame(&mut hop).unwrap();

    // Then: hop RESULT carries both hops and format_telemetry_line is C-shaped
    assert_eq!(typ, MSG_RESULT);
    let result = decode_result_body(&reply).unwrap();
    assert_eq!(result.hdr.status, 0);
    assert_eq!(result.hdr.result_kind, RESULT_LOGITS);
    assert_eq!(decode_logits_payload(&result.payload).unwrap(), logits);
    assert_eq!(result.telemetry.len(), 2);
    assert_eq!(result.telemetry[0].layer_start, 2);
    assert_eq!(result.telemetry[0].layer_end, 3);
    assert_eq!(result.telemetry[0].route_index, 1);
    assert_eq!(result.telemetry[1].layer_start, 0);
    assert_eq!(result.telemetry[1].layer_end, 1);
    assert_eq!(result.telemetry[1].route_index, 0);
    assert_c_shaped_telemetry(11, 0, &result.telemetry[0]);
    assert_c_shaped_telemetry(11, 1, &result.telemetry[1]);

    stop_mid_tx.send(()).unwrap();
    stop_fin_tx.send(()).unwrap();
    drop(hop);
    drop(coord_final);
    drop(coord_middle);
    middle_thread.join().unwrap().unwrap();
    final_thread.join().unwrap().unwrap();
}

#[test]
fn reconnect_local_missing_next_hop_keeps_c_forward_error() {
    // Given: a middle worker whose next hop accepts then closes
    let coord_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let coord_addr = coord_listener.local_addr().unwrap();
    let (middle_listener, middle_port) =
        ds4_dist::open_data_listener(Some("127.0.0.1"), 0).unwrap();
    let next_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let next_port = next_listener.local_addr().unwrap().port();
    let (stop_tx, stop_rx) = mpsc::channel();
    let middle_thread = spawn_reconnect_worker(
        middle_exec(),
        coord_addr,
        middle_listener,
        middle_port,
        stop_rx,
    );
    let coord = accept_hello(&coord_listener);
    let next_thread = thread::spawn(move || {
        let (stream, _) = next_listener.accept().unwrap();
        drop(stream);
    });

    // When: two-hop WORK is forwarded to the closed next hop
    let frame = two_hop_work_frame(middle_port, next_port);
    let mut hop = TcpStream::connect(("127.0.0.1", middle_port)).unwrap();
    tune(&hop);
    hop.write_all(&frame).unwrap();
    let (typ, reply) = read_frame(&mut hop).unwrap();

    // Then: RESULT uses the existing C forward-error string
    assert_eq!(typ, MSG_RESULT);
    let result = decode_result_body(&reply).unwrap();
    assert_ne!(result.hdr.status, 0);
    let msg = String::from_utf8_lossy(&result.payload);
    assert_eq!(msg, ERR_NEXT_CLOSED);
    assert!(!msg.contains("telemetry"));
    next_thread.join().unwrap();

    stop_tx.send(()).unwrap();
    drop(hop);
    drop(coord);
    middle_thread.join().unwrap().unwrap();
}

struct EvalGate {
    started: Mutex<u32>,
    started_cv: Condvar,
    release: Mutex<bool>,
    release_cv: Condvar,
}

impl EvalGate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            started: Mutex::new(0),
            started_cv: Condvar::new(),
            release: Mutex::new(false),
            release_cv: Condvar::new(),
        })
    }

    fn wait_started(&self) {
        let mut n = self.started.lock().expect("eval gate");
        while *n == 0 {
            n = self.started_cv.wait(n).expect("eval gate");
        }
    }

    fn release(&self) {
        *self.release.lock().expect("eval gate") = true;
        self.release_cv.notify_all();
    }
}

struct NotSendExec {
    inner: MockExec,
    gate: Arc<EvalGate>,
    _not_send: PhantomData<*const ()>,
}

impl SliceExec for NotSendExec {
    fn model_id(&self) -> u32 {
        self.inner.model_id()
    }
    fn n_layers(&self) -> u32 {
        self.inner.n_layers()
    }
    fn vocab(&self) -> u32 {
        self.inner.vocab()
    }
    fn ctx_size(&self) -> u32 {
        self.inner.ctx_size()
    }
    fn hidden_values(&self) -> u64 {
        self.inner.hidden_values()
    }
    fn has_output(&self) -> bool {
        self.inner.has_output()
    }
    fn layer_start(&self) -> u32 {
        self.inner.layer_start()
    }
    fn layer_end(&self) -> u32 {
        self.inner.layer_end()
    }
    fn eval(&mut self, req: &WorkRequest) -> Result<WorkOutput, String> {
        {
            let mut n = self.gate.started.lock().expect("eval gate");
            *n += 1;
            self.gate.started_cv.notify_all();
        }
        let mut released = self.gate.release.lock().expect("eval gate");
        while !*released {
            released = self.gate.release_cv.wait(released).expect("eval gate");
        }
        self.inner.eval(req)
    }
}

fn assert_local_prefetch_accepts_not_send<E, S>(
    worker: &mut Worker<E, S>,
    stream: &mut TcpStream,
    queue: &JobQueue<PrefetchJob>,
) -> std::io::Result<()>
where
    E: SliceExec,
    S: ds4_dist::SnapshotStore,
{
    serve_prefetch_local_with(worker, stream, queue)
}

#[test]
fn prefetch_env_unset_enables_local_prefetch() {
    // Given: C default is prefetch on unless DS4_DIST_DISABLE_WORKER_PREFETCH is present
    // When: the disable env is unset
    // Then: local reconnect must take the prefetch path
    assert!(!prefetch_disabled_from(None));
    assert!(ds4_dist::local_prefetch_enabled_from(None));
}

#[test]
fn prefetch_env_set_disables_local_prefetch() {
    // Given: C turns prefetch off when the disable env exists (any value)
    // When: the disable env is present
    // Then: local reconnect must not use prefetch
    assert!(prefetch_disabled_from(Some(std::ffi::OsStr::new("1"))));
    assert!(!ds4_dist::local_prefetch_enabled_from(Some(
        std::ffi::OsStr::new("1")
    )));
}

#[test]
fn serve_prefetch_local_depth_2_queues_second_work_during_not_send_eval() {
    // Given: a !Send exec and a depth-2 local prefetch queue
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (inner, logits) = hop_logits_exec();
    let gate = EvalGate::new();
    let queue = Arc::new(JobQueue::<PrefetchJob>::new(2));
    let frame1 = hop_work_frame_with_id(port, 11);
    let frame2 = hop_work_frame_with_id(port, 12);
    let client_gate = Arc::clone(&gate);
    let client_queue = Arc::clone(&queue);

    let client = thread::spawn(move || {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        tune(&stream);
        stream.write_all(&frame1).unwrap();
        client_gate.wait_started();
        stream.write_all(&frame2).unwrap();
        client_queue.wait_until_queued(1);
        let queued_during_eval = client_queue.queued();
        client_gate.release();
        let first = read_frame(&mut stream).unwrap();
        let second = read_frame(&mut stream).unwrap();
        (queued_during_eval, first, second)
    });

    // When: session-thread prefetch serves both WORKs on a !Send worker
    let mut stream = listener.accept().unwrap().0;
    tune(&stream);
    let mut worker = Worker::new(NotSendExec {
        inner,
        gate,
        _not_send: PhantomData,
    });
    assert_local_prefetch_accepts_not_send(&mut worker, &mut stream, &queue).unwrap();

    // Then: WORK2 was queued while eval1 was still in progress
    let (queued_during_eval, first, second) = client.join().unwrap();
    assert!(
        queued_during_eval >= 1,
        "depth 2 must queue WORK2 during eval1, queued={queued_during_eval}"
    );
    for (typ, reply) in [first, second] {
        assert_eq!(typ, MSG_RESULT);
        let result = decode_result_body(&reply).unwrap();
        assert_eq!(result.hdr.status, 0);
        assert_eq!(result.hdr.result_kind, RESULT_LOGITS);
        assert_eq!(decode_logits_payload(&result.payload).unwrap(), logits);
    }
}
