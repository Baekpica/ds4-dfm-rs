#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn script(dir: &Path, name: &str, body: &str) {
    let path = dir.join(name);
    fs::write(&path, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

impl Fixture {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ds4-perf-test-{}-{nonce}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        script(
            &dir,
            "ds4-bench",
            r#"
if [ "${1:-}" = --help ]; then printf 'NVTX: available\n'; exit 0; fi
if [ "${BENCH_NVTX:-available}" != unknown ]; then printf 'ds4-bench NVTX: %s\n' "${BENCH_NVTX:-available}" >&2; fi
printf '%s %s\n' "$$" "$DS4_QWEN_PLE_CACHE_MB" >> "$BENCH_CALLS"
printf '%s\000' "$@" >> "$BENCH_ARGS"
if [ "${BENCH_FAIL:-0}" = 1 ]; then echo 'intentional benchmark failure' >&2; exit 7; fi
cat "$FIXTURES/bench.csv"
"#,
        );
        script(
            &dir,
            "nsys",
            r#"
case "$1 ${2:-}" in
  'profile --help') echo "--trace cuda nvtx --cuda-graph-trace 'node' --sample --cpuctxsw"; exit 0;;
  'stats --help-reports')
    printf 'The following built-in reports are available:\n'
    printf 'nvtx_gpu_proj_sum -- projection\nnvtx_kern_sum[:base|:mangled] -- kernel\ncuda_gpu_kern_sum[:nvtx-name][:base|:mangled] -- kernel\ncuda_gpu_mem_time_sum -- mem\nnvtx_pushpop_trace -- ranges\ncuda_gpu_trace[:base|:mangled] -- trace\n'
    exit 1;;
  'analyze --help-rules') echo 'The following built-in reports are available:'; echo 'cuda_api_sync[:options] -- sync'; exit 1;;
esac
case "$1" in
  --version) echo 'fake nsys'; exit 0;;
  profile)
    shift
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --output) report=$2; shift 2;;
        --*) shift;;
        *) break;;
      esac
    done
    "$@"
    : > "$report.nsys-rep";;
  stats|analyze)
    if [ "${REPORT_FAIL:-0}" = 1 ]; then echo 'unsupported report fixture' >&2; exit 9; fi
    shift
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --report|--rule) report=$2; shift 2;;
        *) shift;;
      esac
    done
    case "$report" in
      cuda_gpu_kern_sum*) printf 'Name,Total Time (ns),Instances\nglobal_kernel,3500,4\n';;
    esac
    if [ "${BENCH_NVTX:-available}" = unknown ]; then exit 0; fi
    case "$report" in
      nvtx_kern_sum*) cat "$FIXTURES/kernels.csv";;
      nvtx_pushpop_trace) cat "$FIXTURES/ranges.csv";;
      cuda_gpu_trace*) cat "$FIXTURES/gpu-trace.csv";;
      nvtx_gpu_proj_sum) printf 'Range,Total Proj Time (ns),Total Range Time (ns)\n:ds4.prefill,900,1000\n:ds4.decode,800,1000\n';;
      *) echo 'SKIPPED: no data';;
    esac;;
esac
"#,
        );
        script(
            &dir,
            "nvidia-smi",
            r#"
case "${1:-}" in
  --query-compute-apps*) exit 0;;
  --query-gpu*) echo 'Fake GPU, 12.1, fake-driver';;
  *) echo 'CUDA Version: fake';;
esac
"#,
        );
        script(&dir, "ncu", "exit 127");
        script(&dir, "nvcc", "echo 'fake CUDA'");
        Self(dir)
    }

    fn command(&self, out: &str) -> Command {
        self.options(out, &[])
    }

    fn options(&self, out: &str, options: &[&str]) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_ds4-perf"));
        c.current_dir(&self.0)
            .env("PATH", format!("{}:/usr/bin:/bin", self.0.display()))
            .env(
                "FIXTURES",
                Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
            )
            .env("BENCH_CALLS", self.0.join("calls"))
            .env("BENCH_ARGS", self.0.join("args"))
            .env("DS4_QWEN_PLE_CACHE_MB", "1731")
            .env("DS4_PRIVATE_SECRET", "must-not-be-recorded")
            .args([
                "scout",
                "--gpu-helper",
                "/missing-test-helper",
                "--out",
                out,
            ])
            .args(options)
            .args(["--", "./ds4-bench", "a b", "'$(touch BAD)`id`", "--"]);
        c
    }
    fn run(&self, out: &str) -> Output {
        self.command(out).output().unwrap()
    }
}

#[test]
fn fresh_processes_and_artifacts() {
    let f = Fixture::new();
    let result = f.run("run");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let calls = fs::read_to_string(f.0.join("calls")).unwrap();
    let calls: Vec<_> = calls
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .collect();
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0][0], calls[1][0]);
    assert_eq!(calls[0][1], "1731");
    assert_eq!(calls[1][1], "1731");
    let expected = b"a b\0'$(touch BAD)`id`\0--\0";
    assert_eq!(fs::read(f.0.join("args")).unwrap(), expected.repeat(2));
    assert!(!f.0.join("BAD").exists());
    for file in [
        "manifest.txt",
        "command.txt",
        "command.argv",
        "env.txt",
        "bench.stdout",
        "bench.stderr",
        "baseline.csv",
        "trace.nsys-rep",
        "nsys-kernels.csv",
        "nsys-nvtx.csv",
        "nsys-sync.csv",
        "report.txt",
        "normalized-phases.csv",
        "normalized-kernels.csv",
    ] {
        assert!(f.0.join("run").join(file).exists(), "missing {file}");
    }
    let env = fs::read_to_string(f.0.join("run/env.txt")).unwrap();
    assert!(env.contains("1731"));
    assert!(!env.contains("must-not-be-recorded"));
    let report = fs::read_to_string(f.0.join("run/report.txt")).unwrap();
    assert!(report.contains("1502.80 tok/s"));
    assert!(report.contains("GPU coverage    60.0 %"));
    assert!(report.contains("foo<int, 128, bar<float>>"));
    assert!(report.contains("ncu unavailable"));
    assert!(!f.run("run").status.success());
    assert_eq!(
        fs::read_to_string(f.0.join("calls"))
            .unwrap()
            .lines()
            .count(),
        2
    );
}

#[test]
fn failures_preserve_evidence() {
    let f = Fixture::new();
    let result = f.command("failed").env("BENCH_FAIL", "1").output().unwrap();
    assert!(!result.status.success());
    assert!(fs::read_to_string(f.0.join("failed/report.txt"))
        .unwrap()
        .contains("INCOMPLETE SCOUT"));
    assert_eq!(
        fs::read_to_string(f.0.join("calls"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    assert!(fs::read_to_string(f.0.join("failed/status.txt"))
        .unwrap()
        .contains("FAILED"));
    let result = f
        .command("partial")
        .env("REPORT_FAIL", "1")
        .output()
        .unwrap();
    assert!(result.status.success());
    assert!(f.0.join("partial/trace.nsys-rep").exists());
    assert!(fs::read_to_string(f.0.join("partial/report.txt"))
        .unwrap()
        .contains("UNKNOWN"));
    assert!(fs::read_to_string(f.0.join("partial/nsys-kernels.stderr"))
        .unwrap()
        .contains("unsupported report fixture"));
}

#[test]
fn target_capability_is_required() {
    let f = Fixture::new();
    let result = f
        .command("no-nvtx")
        .env("BENCH_NVTX", "unavailable")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("make ds4-bench-perf"));
    assert_eq!(
        fs::read_to_string(f.0.join("calls"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    assert!(!f.0.join("no-nvtx/trace.nsys-rep").exists());
}

#[test]
fn unknown_target_keeps_global_cuda_evidence() {
    let f = Fixture::new();
    let result = f
        .command("legacy")
        .env("BENCH_NVTX", "unknown")
        .output()
        .unwrap();
    assert!(result.status.success());
    let report = fs::read_to_string(f.0.join("legacy/report.txt")).unwrap();
    assert!(report.contains("CUDA-only analysis"));
    assert!(report.contains("global_kernel"));
    assert!(report.contains("ds4.prefill: UNKNOWN"));
    assert!(
        fs::read_to_string(f.0.join("legacy/normalized-kernels.csv"))
            .unwrap()
            .contains(",global_kernel,3500,4")
    );
}

#[test]
fn doctor_checks_selected_benchmark() {
    let f = Fixture::new();
    script(&f.0, "other-bench", "echo 'NVTX: unavailable'");
    let result = Command::new(env!("CARGO_BIN_EXE_ds4-perf"))
        .current_dir(&f.0)
        .env("PATH", format!("{}:/usr/bin:/bin", f.0.display()))
        .args(["doctor", "--bench", "./other-bench"])
        .output()
        .unwrap();
    assert!(result.status.success());
    let report = String::from_utf8_lossy(&result.stdout);
    assert!(report.contains("./other-bench"));
    assert!(report.contains("NVTX             unavailable"));
}

#[test]
fn requested_proof_is_required() {
    let f = Fixture::new();
    let output = f
        .options("no-proof", &["--proof", "--repeats", "3"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "missing requested proof must fail"
    );
    let scout: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("no-proof/scout.json")).unwrap()).unwrap();
    assert_eq!(scout["complete"], false);
}

impl Fixture {
    fn workflow_files(&self) {
        fs::write(self.0.join("model.gguf"), "model fixture").unwrap();
        fs::write(self.0.join("prompt.txt"), "prompt fixture").unwrap();
        let gpu = serde_json::json!({"ordinal":0,"name":"Fixture GPU","uuid":"fixture-uuid","compute_major":12,"compute_minor":1,"driver_version":13030,"total_memory_bytes":1073741824_u64,"free_memory_bytes":536870912_u64,"multiprocessors":48,"warp_size":32,"max_threads_per_block":1024,"max_threads_per_sm":1536,"max_blocks_per_sm":24,"registers_per_sm":65536,"shared_bytes_per_sm":102400,"shared_bytes_per_block":101376,"l2_bytes":25165824,"memory_bus_bits":256,"memory_clock_khz":8533000,"clock_khz":2418000,"unavailable":{}});
        fs::write(self.0.join("gpu.json"), serde_json::to_vec(&gpu).unwrap()).unwrap();
        script(&self.0, "gpu-helper", "cat gpu.json");
        script(
            &self.0,
            "ds4-bench",
            r#"
if [ "${1:-}" = --help ]; then echo 'NVTX: available'; exit 0; fi
printf 'ds4-bench NVTX: available\n' >&2
proof=''
while [ "$#" -gt 0 ]; do
  case "$1" in --dump-frontier-logits-dir) proof=$2; shift;; esac
  shift
done
tps=100
ids='[1,1,1,1,1,1,1,1]'
case "${DS4_QWEN_PREFILL_CHUNK:-256}" in
  512) tps=150;;
  2048) ids='[1,1,1,1,1,1,1,2]';;
  4096) tps=80;;
  8192) echo 'candidate intentionally failed' >&2; exit 7;;
esac
if [ -n "$proof" ] && [ "${DS4_QWEN_PREFILL_CHUNK:-256}" != 16384 ]; then
  mkdir -p "$proof"
  printf '%s\n' '{"source":"ds4-bench","model":"model.gguf","backend":"cuda","quality":false,"quant_bits":2,"prompt_tokens":2048,"frontier_tokens":2048,"prefill_tokens":2048,"ctx":2057,"vocab":3,"argmax_id":1,"argmax_logit":2.0,"logits":[0.0,2.0,1.0]}' > "$proof/frontier_002048.logits.json"
  printf '%s\n' "$ids" > "$proof/tokens-2048.json"
fi
printf 'ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,first_token_sec,kvcache_bytes\n2048,2048,%s,8,20,0.05,1234\n' "$tps"
"#,
        );
        let workload = serde_json::json!({"protocol":"ds4-bench-v1","name":"fixture","family":"qwen","files":{"model":{"path":"model.gguf","sha256":ds4_perf::artifact::hash(&self.0.join("model.gguf")).unwrap()},"prompt":{"path":"prompt.txt","sha256":ds4_perf::artifact::hash(&self.0.join("prompt.txt")).unwrap()}},"shape":{"ctx_tokens":2048,"gen_tokens":8},"cache_state":"fixture warmup process"});
        fs::write(
            self.0.join("workload.json"),
            serde_json::to_vec(&workload).unwrap(),
        )
        .unwrap();
    }

    fn workflow(&self) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_ds4-perf"));
        c.current_dir(&self.0)
            .env("PATH", format!("{}:/usr/bin:/bin", self.0.display()))
            .env(
                "FIXTURES",
                Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
            )
            .env("DS4_QWEN_PREFILL_CHUNK", "256");
        c
    }

    fn source(&self) -> PathBuf {
        self.workflow_files();
        let result = self
            .workflow()
            .args([
                "scout",
                "--out",
                "source",
                "--gpu-helper",
                "./gpu-helper",
                "--proof",
                "--repeats",
                "3",
                "--workload",
                "workload.json",
                "--cache-policy",
                "warmup-then-fresh",
                "--",
                "./ds4-bench",
                "--cuda",
                "-m",
                "model.gguf",
                "--prompt-file",
                "prompt.txt",
                "--ctx-start",
                "2048",
                "--ctx-max",
                "2048",
                "--gen-tokens",
                "8",
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        self.0.join("source/scout.json")
    }

    fn plan(&self, candidates: &[(&str, &str)]) -> PathBuf {
        let plan = serde_json::json!({"candidates":candidates.iter().map(|(name,value)| serde_json::json!({"name":name,"environment":{"DS4_QWEN_PREFILL_CHUNK":value},"reason":"controlled test fixture"})).collect::<Vec<_>>()});
        let path = self.0.join("plan.json");
        fs::write(&path, serde_json::to_vec(&plan).unwrap()).unwrap();
        path
    }
}

#[test]
fn auto_retains_only_proved_gain() {
    let f = Fixture::new();
    let source = f.source();
    let plan = f.plan(&[
        ("fast", "512"),
        ("bad-tokens", "2048"),
        ("failed-process", "8192"),
        ("unreached", "4096"),
    ]);
    let result = f
        .workflow()
        .args(["optimize", "--auto", "--scout"])
        .arg(source)
        .arg("--plan")
        .arg(plan)
        .args(["--out", "auto", "--rounds", "3", "--repeats", "3"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let decision: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("auto/decision.json")).unwrap()).unwrap();
    assert_eq!(decision["complete"], true);
    let data = &decision["data"];
    assert_eq!(data["trials"].as_array().unwrap().len(), 3);
    assert_eq!(data["trials"][0]["accepted"], true);
    assert_eq!(data["trials"][1]["accepted"], false);
    assert_eq!(data["trials"][2]["accepted"], false);
    assert_eq!(
        data["selected"]["environment"]["DS4_QWEN_PREFILL_CHUNK"],
        "512"
    );
    assert_eq!(data["stop_reason"], "round budget reached");
    assert!(f.0.join("auto/candidate-03/warmup.stderr").exists());
    let compare: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("auto/compare-02-before/compare.json")).unwrap())
            .unwrap();
    assert_eq!(compare["data"]["verdict"], "incorrect");
}

#[test]
fn rejects_forged_timings() {
    let f = Fixture::new();
    let source = f.source();
    let original: serde_json::Value = serde_json::from_slice(&fs::read(&source).unwrap()).unwrap();
    for metric in ["prefill_tps", "gen_tps", "first_token_sec", "unhashed"] {
        let mut forged = original.clone();
        if metric == "unhashed" {
            forged["inputs"]
                .as_array_mut()
                .unwrap()
                .retain(|r| r["path"] != "bench.stdout");
        } else {
            forged["data"]["samples"][0]["rows"][0][metric] = 123.0.into();
        }
        fs::write(&source, serde_json::to_vec(&forged).unwrap()).unwrap();
        let out = format!("forged-{metric}");
        f.workflow()
            .args(["compare", "--baseline"])
            .arg(&source)
            .arg("--candidate")
            .arg(&source)
            .args(["--out", &out])
            .output()
            .unwrap();
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(f.0.join(out).join("compare.json")).unwrap()).unwrap();
        assert_eq!(result["data"]["verdict"], "incomparable", "{metric}");
        assert!(result["data"]["reasons"][0]
            .as_str()
            .unwrap()
            .contains("raw benchmark"));
    }
}

#[test]
fn decode_does_not_tune_prefill() {
    let f = Fixture::new();
    let source = f.source();
    let mut data: serde_json::Value = serde_json::from_slice(&fs::read(&source).unwrap()).unwrap();
    for (name, busy, gap, count) in [
        ("idle", 5_000_000, 5_000_000, 1),
        ("fragmented", 9_000_000, 1_000, 1_000),
    ] {
        data["data"]["evidence"]["phases"] = serde_json::json!({
            "ds4.decode": {"name":"ds4.decode", "wall_ns":10_000_000,
                "projected_ns":10_000_000, "busy_ns":busy, "mem_ns":0,
                "gap_ns":gap, "kernels":[{"name":"decode", "total_ns":busy,
                    "count":count}]}
        });
        fs::write(&source, serde_json::to_vec(&data).unwrap()).unwrap();
        let out = format!("decode-{name}");
        let result = f
            .workflow()
            .args(["optimize", "--scout"])
            .arg(&source)
            .args(["--out", &out])
            .output()
            .unwrap();
        assert!(result.status.success());
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(f.0.join(out).join("decision.json")).unwrap())
                .unwrap();
        assert_eq!(result["data"]["candidates"], serde_json::json!([]));
    }
}

#[test]
fn geometry_signals_are_scoped() {
    let f = Fixture::new();
    let source = f.source();
    let mut data: serde_json::Value = serde_json::from_slice(&fs::read(&source).unwrap()).unwrap();
    data["data"]["evidence"]["phases"] = serde_json::json!({
        "ds4.prefill": {"name":"ds4.prefill", "wall_ns":10_000_000,
            "projected_ns":10_000_000, "busy_ns":9_000_000, "mem_ns":0,
            "gap_ns":1_000, "kernels":[{"name":"prefill", "total_ns":9_000_000,
                "count":1}]}
    });
    fs::write(&source, serde_json::to_vec(&data).unwrap()).unwrap();
    for phase in ["ds4.decode", "ds4.prefill"] {
        for kind in ["fit", "ncu"] {
            let data = if kind == "fit" {
                serde_json::json!({"calibration_status":"fixture", "workload_shape_status":"fixture",
                    "bounds":[{"operand_shape":null,"operand_shape_source":"unknown",
                        "launch":{"phase":phase,"kernel":"fixture","grid":[1,1,1],"block":[32,1,1],
                            "registers_per_thread":16,"shared_bytes":0,"instances":1,"total_ns":1000},
                        "resident_blocks_upper":1,"occupancy_upper":0.1,"waves_lower":1,
                        "last_wave_fill":0.1,"block_limits":{}}],"unknown_launches":0,
                    "observed_copy_gb_s":null,"observed_fp32_gflop_s":null,"observed_launch_us":null,
                    "shape":{},"limits":[]})
            } else {
                serde_json::json!({"targets":[{"selection":"fixture", "phase":phase,
                    "kernel":"fixture","captured":true,"launch":{"id":"0","kernel":"fixture",
                        "process":"1","device":"0","grid":"1,1,1","block":"32,1,1"},
                    "metrics":[{"name":"sm__warps_active.avg.pct_of_peak_sustained_active",
                        "unit":"%","value":10.0,"unavailable":null}]}],"limits":[]})
            };
            let path = source.parent().unwrap().join(format!("{kind}.json"));
            let artifact = serde_json::json!({"kind":kind,"schema_version":1,"producer_version":"0.1.1",
                "created_unix":0,"complete":true,"inputs":[],"warnings":[],"data":data});
            fs::write(&path, serde_json::to_vec(&artifact).unwrap()).unwrap();
            let out = format!("geometry-{phase}-{kind}");
            let result = f
                .workflow()
                .args(["optimize", "--scout"])
                .arg(&source)
                .args(["--out", &out])
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let result: serde_json::Value =
                serde_json::from_slice(&fs::read(f.0.join(out).join("decision.json")).unwrap())
                    .unwrap();
            let expected = if phase == "ds4.prefill" { "512" } else { "128" };
            assert_eq!(
                result["data"]["candidates"][0]["environment"]["DS4_QWEN_PREFILL_CHUNK"], expected,
                "{phase} {kind}"
            );
            fs::remove_file(path).unwrap();
        }
    }
}

#[test]
fn rejects_ties_and_regressions() {
    let f = Fixture::new();
    let source = f.source();
    let plan = f.plan(&[("tie", "1024"), ("slow", "4096")]);
    let result = f
        .workflow()
        .args(["optimize", "--auto", "--scout"])
        .arg(&source)
        .arg("--plan")
        .arg(plan)
        .args(["--out", "auto", "--rounds", "2", "--repeats", "3"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let decision: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("auto/decision.json")).unwrap()).unwrap();
    assert_eq!(decision["data"]["selected"], serde_json::Value::Null);
    let result = f
        .workflow()
        .args(["compare", "--regression", "--baseline"])
        .arg(&source)
        .args([
            "--candidate",
            "auto/candidate-02/scout.json",
            "--out",
            "regression",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let comparison: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("regression/compare.json")).unwrap()).unwrap();
    assert_eq!(comparison["data"]["verdict"], "regressed");
    fs::write(
        f.0.join("source/bench-proof/tokens-2048.json"),
        "[2,2,2,2,2,2,2,2]",
    )
    .unwrap();
    let result = f
        .workflow()
        .args(["compare", "--regression", "--baseline"])
        .arg(&source)
        .args([
            "--candidate",
            "auto/candidate-01/scout.json",
            "--out",
            "tampered",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let comparison: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("tampered/compare.json")).unwrap()).unwrap();
    assert_eq!(comparison["data"]["verdict"], "incomparable");
}

#[test]
fn rejects_unused_flags() {
    let f = Fixture::new();
    let result = f
        .options("bad-flags", &["--cupti-sdk", "/unused"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("require --collector cupti"));
}

#[test]
fn records_nsys_replay_identity() {
    let f = Fixture::new();
    assert!(f.run("identity").status.success());
    let scout: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("identity/scout.json")).unwrap()).unwrap();
    assert_eq!(
        scout["data"]["replay"]["collector"]["path"],
        f.0.join("nsys").to_str().unwrap()
    );
}

#[test]
fn rejects_forged_process_scope() {
    let f = Fixture::new();
    let source = f.source();
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&source).unwrap()).unwrap();
    value["data"]["process_scope"]["before"] = serde_json::json!([999999]);
    value["data"]["process_scope"]["verified"] = serde_json::json!(true);
    fs::write(&source, serde_json::to_vec(&value).unwrap()).unwrap();
    let result = f
        .workflow()
        .args(["compare", "--regression", "--baseline"])
        .arg(&source)
        .arg("--candidate")
        .arg(&source)
        .args(["--out", "forged"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let comparison: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("forged/compare.json")).unwrap()).unwrap();
    assert_eq!(comparison["data"]["verdict"], "incomparable");
}

#[test]
fn bounds_ncu_queries() {
    let f = Fixture::new();
    script(
        &f.0,
        "ncu",
        r#"case "${1:-}" in --query-metrics) sleep 3;; *) echo 'fake ncu --launch-count --nvtx';; esac"#,
    );
    let result = f
        .options("query-budget", &["--ncu", "--timeout-seconds", "1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let scout: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("query-budget/scout.json")).unwrap()).unwrap();
    assert_eq!(scout["complete"], false);
    let status = fs::read_to_string(f.0.join("query-budget/ncu/query.status.txt")).unwrap();
    assert!(status.contains("deadline"), "{status}");
}

#[test]
fn bounds_decision_publication() {
    let f = Fixture::new();
    let source = f.source();
    let plan = f.plan(&[("large-reason", "512")]);
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&plan).unwrap()).unwrap();
    value["candidates"][0]["reason"] = serde_json::json!("x".repeat(1100000));
    fs::write(&plan, serde_json::to_vec(&value).unwrap()).unwrap();
    let result = f
        .workflow()
        .args(["optimize", "--scout"])
        .arg(source)
        .arg("--plan")
        .arg(plan)
        .args(["--out", "large-decision", "--max-output-mib", "1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let decision: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("large-decision/decision.json")).unwrap())
            .unwrap();
    assert_eq!(decision["complete"], false);
    assert!(decision["data"]["stop_reason"]
        .as_str()
        .unwrap()
        .contains("budget"));
}

#[test]
fn failed_probe_is_partial() {
    let f = Fixture::new();
    script(&f.0, "ps", "exit 1");
    let result = f.run("probe-failed");
    assert!(!result.status.success());
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(f.0.join("probe-failed/scout.json")).unwrap()).unwrap();
    assert_eq!(value["complete"], false);
}

#[test]
fn rejects_unbound_device() {
    let f = Fixture::new();
    let result = f
        .options("device-mismatch", &["--device", "1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("CUDA_VISIBLE_DEVICES"));
}

#[test]
fn partial_fit_is_incomplete() {
    let f = Fixture::new();
    f.workflow_files();
    let result = f
        .workflow()
        .args([
            "scout",
            "--out",
            "partial-fit",
            "--fit",
            "--gpu-helper",
            "./gpu-helper",
            "--workload",
            "workload.json",
            "--",
            "./ds4-bench",
            "--cuda",
            "-m",
            "model.gguf",
            "--prompt-file",
            "prompt.txt",
            "--ctx-start",
            "2048",
            "--ctx-max",
            "2048",
            "--gen-tokens",
            "8",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    for name in ["fit.json", "scout.json"] {
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(f.0.join("partial-fit").join(name)).unwrap()).unwrap();
        assert_eq!(value["complete"], false);
    }
}
