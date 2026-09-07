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
                "--out",
                out,
                "--",
                "./ds4-bench",
                "a b",
                "'$(touch BAD)`id`",
                "--",
            ]);
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
