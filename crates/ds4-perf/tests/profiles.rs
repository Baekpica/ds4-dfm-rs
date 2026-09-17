use serde_json::{json, Value};
use std::{fs, path::PathBuf, process::Command};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ds4-profile-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn controls_follow_runtime_capabilities() {
    let f = Fixture::new();
    let plan = json!({"family":"family-without-perf-name-table", "requested":{}, "effective":{"native_chunk":1024,"sched_chunk":256,"sched_chunk_live":512}, "qualified":{}, "issues":[],
        "controls":{"native_prefill_env":"DS4_EXAONE_PREFILL_CHUNK", "scheduler_chunks":[256,512,1024],"prefix_reuse":"partial","banks":"persistent","mtp":"none","disk":"qualified"}});
    fs::write(f.0.join("plan.json"), serde_json::to_vec(&plan).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ds4-perf"))
        .args(["serving-controls", "--plan"])
        .arg(f.0.join("plan.json"))
        .arg("--out")
        .arg(f.0.join("controls"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let artifact: Value =
        serde_json::from_slice(&fs::read(f.0.join("controls/controls.json")).unwrap()).unwrap();
    let candidates = artifact["data"]["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 4);
    for candidate in candidates {
        assert_eq!(candidate["qualified"], false);
        let options = candidate["arguments"].as_object().unwrap();
        assert_eq!(options.len(), 1);
        let (key, value) = options.iter().next().unwrap();
        assert!(["--prefill-chunk", "--prefill-chunk-live"].contains(&key.as_str()));
        assert!([256, 512, 1024].contains(&value.as_u64().unwrap()));
    }
}

#[test]
fn profile_refuses_unmeasured_and_override_arguments() {
    let f = Fixture::new();
    fs::write(
        f.0.join("selection.json"),
        br#"{"workload":"missing.json","candidates":[{"name":"unmeasured","runs":[]}]}"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ds4-perf"))
        .args(["serving-profile", "--plan"])
        .arg(f.0.join("selection.json"))
        .arg("--out")
        .arg(f.0.join("selected"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("at least one run"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(env!("CARGO_BIN_EXE_ds4-perf"))
        .args([
            "apply-profile",
            "--profile",
            "missing.json",
            "--out",
            "missing",
            "--",
            "--max-seqs",
            "64",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unexpected argument"));
}
