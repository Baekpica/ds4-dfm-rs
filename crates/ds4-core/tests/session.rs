//! C↔Rust session host ledger: rewrite, prefix, plan, rewind, generation.

use std::path::PathBuf;
use std::process::Command;

use ds4_core::session_dump_cmd;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_SESSION_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/session_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/session_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_cmd(args: &[&str]) -> String {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run session_c_oracle");
    assert!(
        out.status.success(),
        "oracle {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

fn assert_cmd(cmd: &str, args: &[&str]) {
    let mut c_args = vec![cmd];
    c_args.extend_from_slice(args);
    let c = c_cmd(&c_args);
    let rust = session_dump_cmd(cmd, args);
    assert_eq!(rust, c, "mismatch {cmd} {args:?}");
}

#[test]
fn rewrite_requires_rebuild_matches_c() {
    assert_cmd("rewrite", &["19296", "19290", "19081"]);
    assert_cmd("rewrite", &["1024", "1030", "1000"]);
    assert_cmd("rewrite", &["1024", "900", "900"]);
    assert_cmd("rewrite", &["1024", "1024", "1024"]);
    assert_cmd("rewrite", &["1024", "1100", "1024"]);
    assert_cmd("rewrite", &["-1", "4", "0"]);
    assert_cmd("rewrite", &["4", "4", "5"]);
}

#[test]
fn common_prefix_and_rewrite_from_match_c() {
    assert_cmd("prefix", &["1,2,3", "1,2,3,4"]);
    assert_cmd("prefix", &["1,2,9", "1,2,3"]);
    assert_cmd("prefix", &["-", "1,2"]);
    assert_cmd("prefix", &["1,2", "-"]);
    assert_cmd("rewrite-from", &["16", "1,2,3", "1,2,3,4", "3"]);
    assert_cmd("rewrite-from", &["16", "1,2,3", "1,2,9", "2"]);
    assert_cmd("rewrite-from", &["16", "1,2,3", "1,2,3", "1"]);
    assert_cmd("rewrite-from", &["4", "1,2,3", "1,2,3", "3"]);
    assert_cmd("rewrite-from", &["16", "1,2,3", "9,2,3", "2"]);
}

#[test]
fn sync_plans_match_c() {
    assert_cmd(
        "plan",
        &["deepseek4", "cuda", "32", "8", "1", "1", "1,2,3;1,2,3,4"],
    );
    assert_cmd(
        "plan",
        &["deepseek4", "cuda", "32", "8", "1", "1", "1,2,3;9,8,7"],
    );
    assert_cmd(
        "plan",
        &["deepseek4", "cuda", "32", "8", "1", "1", "1,2,3;1,2,3"],
    );
    assert_cmd(
        "plan",
        &["deepseek4", "cuda", "8", "8", "0", "1", "-;1,2,3,4,5,6,7"],
    );
    assert_cmd("plan", &["qwen4exp", "cuda", "8", "8", "0", "1", "-;n:8"]);
    assert_cmd(
        "plan",
        &["deepseek4", "cuda", "16384", "9000", "0", "1", "-;1"],
    );
    assert_cmd(
        "plan",
        &["deepseek4", "cuda", "16384", "9000", "0", "1", "-;n:9000"],
    );
    assert_cmd("plan", &["motif3", "cuda", "32", "8", "1", "1", "1,2;9,8"]);
    assert_cmd(
        "plan",
        &[
            "dots3-note",
            "cuda",
            "32",
            "4",
            "1",
            "1",
            "1,2,3,4,5;1,2,3,4,5,6",
        ],
    );
    assert_cmd(
        "plan",
        &["solar-open2", "cuda", "32", "8", "1", "0", "1,2;1,2,3"],
    );
    assert_cmd(
        "plan",
        &["solar-open2", "cuda", "32", "8", "1", "1", "1,2;1,2,3"],
    );
    assert_cmd(
        "plan",
        &[
            "exaone-moe",
            "cuda",
            "32",
            "8",
            "1",
            "1",
            "1,2,3,4,5;1,2,3,4,5",
        ],
    );
    assert_cmd(
        "plan",
        &["exaone-moe", "cuda", "32", "8", "1", "1", "1,2,3,4,5;1,2,9"],
    );
    assert_cmd(
        "plan",
        &["exaone-moe", "cuda", "32", "8", "1", "1", "1,2,3,4,5;1,2,9"],
    );
    assert_cmd("plan", &["deepseek4", "cpu", "8", "8", "0", "1", "-;-"]);
}

#[test]
fn rewind_invalidate_create_match_c() {
    assert_cmd("rewind", &["16", "5", "3", "0"]);
    assert_cmd("rewind", &["16", "5", "5", "0"]);
    assert_cmd("rewind", &["16", "5", "0", "0"]);
    assert_cmd("rewind", &["16", "5", "-1", "0"]);
    assert_cmd("rewind", &["16", "5", "9", "0"]);
    assert_cmd("rewind", &["16", "5", "3", "1"]);
    assert_cmd("rewind", &["16", "5", "5", "1"]);
    assert_cmd("invalidate", &[]);
    assert_cmd("create", &["2048"]);
}
