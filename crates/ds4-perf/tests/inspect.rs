use std::{fs, process::Command};

#[test]
fn inspect_without_cuda() {
    let root = std::env::temp_dir().join(format!("ds4-inspect-{}", std::process::id()));
    let out = root.join("machine");
    fs::create_dir(&root).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_ds4-perf"))
        .args(["inspect", "--out"])
        .arg(&out)
        .arg("--gpu-helper")
        .arg(root.join("not-installed"))
        .env("PATH", &root)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let machine: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("machine.json")).unwrap()).unwrap();
    assert_eq!(machine["kind"], "machine");
    assert_eq!(machine["schema_version"], 1);
    assert_eq!(machine["data"]["gpu"], serde_json::Value::Null);
    assert!(!machine["warnings"].as_array().unwrap().is_empty());
    // Evidence directories are immutable, including an incomplete inspection.
    let again = Command::new(env!("CARGO_BIN_EXE_ds4-perf"))
        .args(["inspect", "--out"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(!again.status.success());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn four_command_help() {
    for (command, flags) in [
        ("inspect", vec!["--calibrate"]),
        ("scout", vec!["--fit", "--ncu", "--collector"]),
        ("compare", vec!["--baseline", "--regression"]),
        ("optimize", vec!["--auto"]),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_ds4-perf"))
            .args([command, "--help"])
            .output()
            .unwrap();
        assert!(result.status.success(), "missing {command} help");
        let text = String::from_utf8_lossy(&result.stdout);
        for flag in flags {
            assert!(text.contains(flag), "missing {command} {flag}");
        }
    }
}
