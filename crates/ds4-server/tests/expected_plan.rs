//! Exercise the production CLI without opening a device or loading weights.
use serde_json::{json, Value};
use std::{fs, process::Command};

#[test]
#[ignore = "needs DS4_EXPECT_PLAN_SERVER, DS4_EXPECT_PLAN_MODEL and memory for the preflight quote"]
fn preflight_checks_policy_without_post_open_quote() {
    let server = std::env::var("DS4_EXPECT_PLAN_SERVER")
        .expect("set DS4_EXPECT_PLAN_SERVER to the frozen native server binary");
    let model = std::env::var("DS4_EXPECT_PLAN_MODEL")
        .expect("set DS4_EXPECT_PLAN_MODEL to a validated local GGUF");
    let command = || {
        let mut command = Command::new(&server);
        for (name, _) in std::env::vars_os() {
            if name.to_string_lossy().starts_with("DS4_") {
                command.env_remove(name);
            }
        }
        command.args([
            "--model",
            &model,
            "--check-config",
            "--backend",
            "cpu",
            "--max-seqs",
            "1",
            "--ctx",
            "512",
            "--mtp-mode",
            "off",
            "--prefix-reuse",
            "off",
        ]);
        command
    };
    let baseline = command().output().unwrap();
    assert!(
        baseline.status.success(),
        "{}",
        String::from_utf8_lossy(&baseline.stderr)
    );
    let plan: Value = serde_json::from_slice(&baseline.stdout).unwrap();
    let mut expected = json!({
        "family":plan["family"], "backend":plan["requested"]["backend"],
        "effective":plan["effective"], "qualified":plan["qualified"],
        "controls":plan["controls"], "post_open_quote":{"shared_weights":0}
    });
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let guard =
        std::env::temp_dir().join(format!("ds4-preflight-{}-{nonce}.json", std::process::id()));
    fs::write(&guard, serde_json::to_vec(&expected).unwrap()).unwrap();
    let accepted = command().arg("--expect-plan").arg(&guard).output().unwrap();
    expected["effective"]["ctx"] = json!(513);
    fs::write(&guard, serde_json::to_vec(&expected).unwrap()).unwrap();
    let rejected = command().arg("--expect-plan").arg(&guard).output().unwrap();
    fs::remove_file(guard).unwrap();
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("mismatch: effective"));
}
