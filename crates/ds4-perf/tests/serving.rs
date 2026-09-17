#![cfg(target_os = "linux")]

use serde_json::{json, Value};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpListener,
    path::PathBuf,
    process::Command,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

struct Fixture {
    root: PathBuf,
    url: String,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Fixture {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let root = std::env::temp_dir().join(format!("ds4-serving-{}-{port}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let finished = stop.clone();
        let worker = thread::spawn(move || {
            let completed = Arc::new(AtomicUsize::new(0));
            let mut handlers = Vec::new();
            while !finished.load(Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                };
                let completed = completed.clone();
                handlers.push(thread::spawn(move || {
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let is_stats = line.starts_with("GET /v1/stats ");
                let mut length = 0;
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse().unwrap();
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                if is_stats {
                    let completed = completed.load(Ordering::Relaxed);
                    let reuse = ["cold", "exact", "partial"][completed.saturating_sub(1).min(2)];
                    let body = json!({
                        "routes": {"chat_continuous": completed}, "queue_depth": 0, "clients": 1,
                        "serving": {
                            "family": "qwen4exp", "requested": {},
                            "effective": {"prefix_reuse": "partial", "ctx": 4096, "max_seqs": 2},
                            "qualified": {}, "issues": []
                        },
                        "last_request": {"effective_lane": "continuous", "reuse_kind": reuse,
                            "speculation_active": false, "fallback_reason": null}
                    })
                    .to_string();
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .unwrap();
                    return;
                }
                let request: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(request["stream"], true);
                stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
                stream
                    .write_all(b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n")
                    .unwrap();
                if request["messages"][0]["content"] == "tool-call" {
                    stream.write_all(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-a\",\"type\":\"function\",\"function\":{\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n").unwrap();
                    stream.write_all(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Seoul\\\"}\"}}]}}]}\n\n").unwrap();
                } else if request["messages"][0]["content"] == "overlap-decode" {
                    for _ in 0..8 {
                        thread::sleep(Duration::from_millis(40));
                        stream.write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n").unwrap();
                    }
                } else {
                thread::sleep(Duration::from_millis(120));
                // The collector must handle an SSE event split across writes.
                stream
                    .write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"ans")
                    .unwrap();
                thread::sleep(Duration::from_millis(10));
                stream.write_all(b"wer\"}}]}\n\n").unwrap();
                thread::sleep(Duration::from_millis(30));
                }
                    let finish = if request["messages"][0]["content"] == "tool-call" { "tool_calls" } else { "stop" };
                    write!(stream, "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"{finish}\"}}]}}\n\n").unwrap();
                    if request["stream_options"]["include_usage"] == true {
                        let decoding = request["messages"][0]["content"] == "overlap-decode";
                        let usage = json!({"choices": [], "usage": {"prompt_tokens": if decoding {32} else {8192}, "prompt_tokens_details":{"cached_tokens":0}, "completion_tokens": if decoding {8} else {1}}});
                        write!(stream, "data: {usage}\n\n").unwrap();
                    }
                    stream.write_all(b"data: [DONE]\n\n").unwrap();
                completed.fetch_add(1, Ordering::Relaxed);
                }));
            }
            for handler in handlers {
                handler.join().unwrap();
            }
        });
        Self {
            root,
            url: format!("http://127.0.0.1:{port}"),
            stop,
            worker: Some(worker),
        }
    }

    fn workload(&self) -> Value {
        json!({
            "protocol": "ds4-serving-v1", "name": "three-conversation-paths", "family": "qwen4exp",
            "cases": (["kv_cold_prefill", "warm_append", "partial_branch"].iter().zip(["cold", "exact", "partial"])
                .enumerate().map(|(i, (scenario, reuse))| json!({
                    "name": format!("case-{i}"), "scenario": scenario,
                    "request": {"model": "fixture", "messages": [{"role":"user", "content": format!("turn {i}")}], "stream": true, "temperature": 0, "max_tokens": 8},
                    "expect": {"content": "answer", "finish_reason": "stop", "reuse_kind": reuse, "effective_lane": "continuous", "speculation_active": false, "fallback_reason": null},
                    "limits": {"ttft_ms": 2000, "total_ms": 3000, "min_host_available_bytes": 1}
                })).collect::<Vec<_>>())
        })
    }

    fn run(&self, workload: &Value) -> std::process::Output {
        self.run_options(workload, &[])
    }

    fn run_options(&self, workload: &Value, options: &[&str]) -> std::process::Output {
        fs::write(
            self.root.join("workload.json"),
            serde_json::to_vec(workload).unwrap(),
        )
        .unwrap();
        Command::new(env!("CARGO_BIN_EXE_ds4-perf"))
            .args(["serving", "--url", &self.url, "--workload"])
            .arg(self.root.join("workload.json"))
            .arg("--out")
            .arg(self.root.join("run"))
            .args(["--timeout-seconds", "5"])
            .args(options)
            .output()
            .unwrap()
    }
}

#[test]
fn pins_identity_and_preserves_repeat_cache() {
    let fixture = Fixture::new();
    let mut workload = fixture.workload();
    workload["cases"].as_array_mut().unwrap().truncate(1);
    let model = fixture.root.join("model.fixture");
    fs::write(&model, b"model identity fixture").unwrap();
    workload["inputs"] = json!({"model": {"path": "model.fixture", "sha256": ds4_perf::artifact::hash(&model).unwrap()}});
    let pid = std::process::id().to_string();
    let output = fixture.run_options(&workload, &["--repeats", "2", "--server-pid", &pid]);
    // Replaying a cold case must expose that the second request actually reused KV.
    assert!(!output.status.success());
    let artifact: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("run/serving.json")).unwrap()).unwrap();
    assert_eq!(artifact["complete"], true, "{artifact}");
    assert_eq!(artifact["data"]["requested_repeats"], 2);
    assert_eq!(
        artifact["data"]["identity"]["server"]["pid"],
        std::process::id()
    );
    assert_eq!(artifact["data"]["cases"][0]["repeat"], 0);
    assert_eq!(artifact["data"]["cases"][1]["repeat"], 1);
    assert!(artifact["data"]["cases"][1]["failures"]
        .to_string()
        .contains("reuse_kind"));
}

#[test]
fn decode_progress_during_peer_prefill() {
    let fixture = Fixture::new();
    let mut workload = fixture.workload();
    let case = &workload["cases"][0];
    let mut decode = json!({"request": case["request"], "expect": {"content": "aaaaaaaa", "finish_reason": "stop"}, "limits": case["limits"]});
    decode["request"]["messages"][0]["content"] = json!("overlap-decode");
    decode["request"]["stream_options"] = json!({"include_usage":true});
    let mut prefill = json!({"request": case["request"], "expect": {"content": "answer", "finish_reason": "stop"}, "limits": case["limits"]});
    prefill["request"]["stream_options"] = json!({"include_usage":true});
    workload["cases"] = json!([]);
    workload["overlaps"] = json!([{"name": "decode-peer-prefill", "decode": decode, "prefill": prefill,
        "min_decode_events_during_prefill": 2, "min_prefill_tokens": 4096, "max_decode_gap_ms": 100}]);
    let output = fixture.run(&workload);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let artifact: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("run/serving.json")).unwrap()).unwrap();
    let overlap = &artifact["data"]["overlaps"][0];
    assert!(overlap["decode_events_during_prefill"].as_u64().unwrap() >= 2);
    assert!(overlap["max_decode_gap_ms"].as_f64().unwrap() <= 100.0);
    assert_eq!(artifact["data"]["passed"], true);
    assert_eq!(overlap["prefill_computed_tokens"], 8192);
    assert_eq!(overlap["decode_committed_tokens"], 8);
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn captures_serving_latency_and_reuse() {
    let fixture = Fixture::new();
    let output = fixture.run(&fixture.workload());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let artifact: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("run/serving.json")).unwrap()).unwrap();
    assert_eq!(artifact["kind"], "serving");
    assert_eq!(artifact["complete"], true);
    assert_eq!(artifact["data"]["passed"], true);
    let cases = artifact["data"]["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        assert!(case["ttft_ms"].as_f64().unwrap() >= 120.0);
        assert!(case["total_ms"].as_f64().unwrap() >= 150.0);
        assert_eq!(case["content"], "answer");
        assert!(case["host_min_available_bytes"]
            .as_u64()
            .is_some_and(|n| n > 0));
        assert!(case["failures"].as_array().unwrap().is_empty());
    }
    for file in [
        "request.json",
        "response.sse",
        "events.json",
        "before.json",
        "after.json",
        "memory.json",
    ] {
        assert!(fixture.root.join("run/case-000").join(file).is_file());
    }
    assert!(!fixture.run(&fixture.workload()).status.success());
}

#[test]
fn refuses_incorrect_or_missed_reuse() {
    for (key, value, expected) in [
        ("content", "wrong", "content"),
        ("finish_reason", "length", "finish_reason"),
        ("reuse_kind", "exact", "reuse_kind"),
    ] {
        let fixture = Fixture::new();
        let mut workload = fixture.workload();
        workload["cases"].as_array_mut().unwrap().truncate(1);
        workload["cases"][0]["expect"][key] = json!(value);
        if key == "reuse_kind" {
            workload["cases"][0]["scenario"] = json!("warm_append");
        }
        let output = fixture.run(&workload);
        assert!(!output.status.success());
        let artifact: Value =
            serde_json::from_slice(&fs::read(fixture.root.join("run/serving.json")).unwrap())
                .unwrap();
        assert_eq!(artifact["complete"], true);
        assert_eq!(artifact["data"]["passed"], false);
        assert!(artifact["data"]["cases"][0]["failures"]
            .to_string()
            .contains(expected));
    }
}

#[test]
fn rejects_latency_and_memory_failure() {
    let fixture = Fixture::new();
    let mut workload = fixture.workload();
    workload["cases"].as_array_mut().unwrap().truncate(1);
    workload["cases"][0]["limits"]["ttft_ms"] = json!(1);
    workload["cases"][0]["limits"]["min_host_available_bytes"] = json!(u64::MAX);
    assert!(!fixture.run(&workload).status.success());
    let artifact: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("run/serving.json")).unwrap()).unwrap();
    let failures = artifact["data"]["cases"][0]["failures"].to_string();
    assert!(failures.contains("TTFT"), "{failures}");
    assert!(failures.contains("memory"), "{failures}");
}

#[test]
fn measures_and_checks_generated_tool_calls() {
    for city in ["Seoul", "Busan"] {
        let fixture = Fixture::new();
        let mut workload = fixture.workload();
        workload["cases"].as_array_mut().unwrap().truncate(1);
        let case = &mut workload["cases"][0];
        case["scenario"] = json!("tool");
        case["request"]["messages"][0]["content"] = json!("tool-call");
        case["request"]["tools"] = json!([{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]);
        case["expect"]["content"] = json!("");
        case["expect"]["finish_reason"] = json!("tool_calls");
        case["expect"]["tool_calls"] = json!([{"name":"weather","arguments":{"city":city}}]);
        let output = fixture.run(&workload);
        assert_eq!(
            output.status.success(),
            city == "Seoul",
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let evidence: Value =
            serde_json::from_slice(&fs::read(fixture.root.join("run/serving.json")).unwrap())
                .unwrap();
        assert_eq!(
            evidence["data"]["cases"][0]["tool_calls"][0]["arguments"]["city"],
            "Seoul"
        );
        assert!(evidence["data"]["gpu_windows"][0]["samples"]
            .as_array()
            .is_some_and(|samples| !samples.is_empty()));
    }
}

#[test]
fn cache_usage_minimum_cannot_pass_on_a_cold_trace() {
    let fixture = Fixture::new();
    let mut workload = fixture.workload();
    workload["cases"].as_array_mut().unwrap().truncate(1);
    workload["cases"][0]["request"]["stream_options"] = json!({"include_usage":true});
    workload["cases"][0]["expect"]["min_cached_tokens"] = json!(1);
    let output = fixture.run(&workload);
    assert!(!output.status.success());
    let evidence: Value =
        serde_json::from_slice(&fs::read(fixture.root.join("run/serving.json")).unwrap()).unwrap();
    assert_eq!(evidence["data"]["cases"][0]["cached_tokens"], 0);
    assert!(evidence["data"]["cases"][0]["failures"]
        .as_array()
        .unwrap()
        .iter()
        .any(|failure| failure.as_str().unwrap().contains("cached token")));
}

// A separate process makes restart provenance testable without any model.
#[test]
fn restart_peer() {
    let Some(root) = std::env::var_os("DS4_PERF_TEST_PEER") else {
        return;
    };
    let root = PathBuf::from(root);
    let address = fs::read_to_string(root.join("address")).unwrap();
    let listener = TcpListener::bind(address.trim()).unwrap();
    let restored = root.join("seeded").exists();
    let mut completed = 0;
    fs::write(root.join("ready"), b"ready").unwrap();
    for connection in listener.incoming() {
        let mut stream = connection.unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let stats = line.starts_with("GET /v1/stats ");
        let mut length = 0;
        loop {
            line.clear();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                length = value.trim().parse().unwrap();
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        if stats {
            let body = json!({"routes":{"chat":completed},"queue_depth":0,"clients":1,
                "serving":{"family":"qwen4exp","requested":{},"effective":{"disk":true,"max_seqs":1},"qualified":{},"issues":[]},
                "last_request":{"reuse_kind":if restored {"exact"} else {"cold"},"effective_lane":"serial","speculation_active":false,"fallback_reason":null}}).to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            continue;
        }
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n").unwrap();
        let usage = json!({"choices":[],"usage":{"prompt_tokens":32,"prompt_tokens_details":{"cached_tokens":if restored {16} else {0}},"completion_tokens":1}});
        write!(stream, "data: {usage}\n\ndata: [DONE]\n\n").unwrap();
        fs::write(root.join("seeded"), b"checkpoint").unwrap();
        completed += 1;
    }
}

struct Peer(std::process::Child);
impl Drop for Peer {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn start_peer(root: &std::path::Path) -> Peer {
    let _ = fs::remove_file(root.join("ready"));
    let child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "restart_peer", "--nocapture"])
        .env("DS4_PERF_TEST_PEER", root)
        .current_dir(root)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let peer = Peer(child);
    for _ in 0..100 {
        if root.join("ready").exists() {
            return peer;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("restart peer did not start");
}

#[test]
fn links_fresh_process_disk_restore() {
    let fixture = Fixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    fs::write(fixture.root.join("address"), address.to_string()).unwrap();
    let mut workload = fixture.workload();
    workload["cases"].as_array_mut().unwrap().truncate(1);
    workload["cases"][0]["expect"]["effective_lane"] = json!("serial");
    workload["cases"][0]["request"]["stream_options"] = json!({"include_usage":true});
    let collect = |name: &str, workload: &Value, pid: u32| {
        let file = fixture.root.join(format!("{name}.json"));
        fs::write(&file, serde_json::to_vec(workload).unwrap()).unwrap();
        Command::new(env!("CARGO_BIN_EXE_ds4-perf"))
            .args(["serving", "--url"])
            .arg(format!("http://{address}"))
            .arg("--workload")
            .arg(file)
            .arg("--out")
            .arg(fixture.root.join(name))
            .arg("--server-pid")
            .arg(pid.to_string())
            .args(["--timeout-seconds", "5"])
            .output()
            .unwrap()
    };
    let peer = start_peer(&fixture.root);
    let seed = collect("seed", &workload, peer.0.id());
    assert!(
        seed.status.success(),
        "{}",
        String::from_utf8_lossy(&seed.stderr)
    );
    drop(peer);
    // Same executable, argv and working directory; only the process changes.
    let peer = start_peer(&fixture.root);
    workload["restart_from"] = json!(fixture.root.join("seed/serving.json"));
    workload["cases"][0]["scenario"] = json!("restart_restore");
    workload["cases"][0]["expect"]["reuse_kind"] = json!("exact");
    workload["cases"][0]["expect"]["min_cached_tokens"] = json!(16);
    let restored = collect("restored", &workload, peer.0.id());
    assert!(
        restored.status.success(),
        "{}",
        String::from_utf8_lossy(&restored.stderr)
    );
    let second = collect("already-used", &workload, peer.0.id());
    assert!(!second.status.success());
    assert!(String::from_utf8_lossy(&second.stderr).contains("first request"));
}
