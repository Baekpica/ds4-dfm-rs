//! C↔Rust corrective-retry tapes (suffix / repair / decide). No model.

use ds4_server::{
    retry::build_invalid_tool_error_suffix, retry_dump_script, ChatFormat, ThinkMode,
};
use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_RETRY_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/retry_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/retry_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_str(name: &str) -> String {
    let out = Command::new(require_oracle())
        .arg(name)
        .output()
        .expect("run retry_c_oracle");
    assert!(
        out.status.success(),
        "oracle {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

fn assert_script(name: &str) {
    let rust = retry_dump_script(name);
    let c = c_str(name);
    if rust != c {
        panic!("{name} mismatch\n--- rust ---\n{rust}\n--- c ---\n{c}");
    }
}

#[test]
fn terminal_finish_keeps_cause() {
    assert_eq!(ds4_server::retry::terminal_finish("stop"), "stop");
    assert_eq!(
        ds4_server::retry::terminal_finish("tool_calls"),
        "tool_calls"
    );
    assert_eq!(ds4_server::retry::terminal_finish("length"), "length");
    assert_eq!(ds4_server::retry::terminal_finish("error"), "error");
    assert_eq!(ds4_server::retry::terminal_finish("other"), "stop");
}

#[test]
fn suffix_and_system_match_c() {
    for name in [
        "dsml-think",
        "dsml-nothink",
        "solar-think",
        "system-dsml",
        "system-solar",
    ] {
        assert_script(name);
    }
    let dsml = retry_dump_script("dsml-think");
    assert!(dsml.starts_with("</think><｜end▁of▁sentence｜><｜User｜><tool_result>"));
    assert!(dsml.contains("Tool error: invalid DSML tool call: missing invoke name"));
    assert!(dsml.contains("System prompt reminder:\n## Tools\nschema\n\nSystem rule"));
    assert!(dsml.contains("</tool_result><｜Assistant｜><think>"));
    assert!(!dsml.contains("<｜User｜>Hi"));

    let solar = retry_dump_script("solar-think");
    assert!(!solar.contains("DSML"));
    assert!(solar.contains("invalid Solar tool call"));
    assert!(solar.contains("System prompt reminder:\n## System Prompt\n\nStay precise."));
}

#[test]
fn repair_and_decide_match_c() {
    for name in [
        "repair-dsml",
        "repair-solar",
        "repair-dsml-none",
        "decide-unterminated-stop",
        "decide-unterminated-length",
        "decide-parse-retry",
        "decide-parse-motif",
    ] {
        assert_script(name);
    }
    assert_eq!(retry_dump_script("repair-dsml-none"), "NONE");
    assert_eq!(
        retry_dump_script("decide-unterminated-stop").trim(),
        "retry-unterminated"
    );
    assert_eq!(
        retry_dump_script("decide-unterminated-length").trim(),
        "none"
    );
    assert_eq!(retry_dump_script("decide-parse-retry").trim(), "true");
    assert_eq!(retry_dump_script("decide-parse-motif").trim(), "false");
}

#[test]
fn qwen_invalid_tool_retry_uses_native_protocol() {
    let suffix = build_invalid_tool_error_suffix(
        ChatFormat::Qwen4Exp,
        ThinkMode::None,
        false,
        b"<|im_start|>system\nStay precise.<|im_end|>\n",
        "missing function",
    );
    let suffix = String::from_utf8(suffix).unwrap();
    assert!(suffix.starts_with(
        "<|im_end|>\n<|im_start|>user\n<tool_response>\nTool error: invalid Qwen tool call: missing function"
    ));
    assert!(suffix.contains("System prompt reminder:\nStay precise."));
    assert!(suffix
        .ends_with("\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}
