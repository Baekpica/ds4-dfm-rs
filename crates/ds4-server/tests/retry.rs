//! C↔Rust corrective-retry tapes (suffix / repair / decide). No model.

use ds4_server::{
    retry::build_invalid_tool_error_suffix, retry_dump_script, ChatFormat, ModelSyntax, ThinkMode,
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
        "repair-dsml-param-open",
        "repair-dsml-invoke-open",
        "repair-dsml-spaced",
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

/// The reference's repair self-test, ported case by case
/// (ds4_server.c:19322 `test_dsml_repair_produces_parseable_calls` and the
/// balance cases around it).  Only a missing wrapper is repairable: an
/// unfinished invocation can still be missing arguments and an unfinished value
/// can be a truncated command, so closing either "would invent an executable
/// action" (:6697-6700).  A repair that closes them was the deviation G8
/// recorded.
#[test]
fn dsml_repair_matches_the_reference_balance_contract() {
    use ds4_server::retry::try_repair_dsml;
    use ds4_server::tools::{
        DSML_INVOKE_END, DSML_INVOKE_END_SHORT, DSML_INVOKE_START, DSML_INVOKE_START_SHORT,
        DSML_PARAM_END, DSML_PARAM_END_SHORT, DSML_PARAM_START, DSML_PARAM_START_SHORT,
        DSML_TOOL_CALLS_END, DSML_TOOL_CALLS_END_SHORT, DSML_TOOL_CALLS_START,
        DSML_TOOL_CALLS_START_SHORT,
    };
    let repair =
        |text: String| try_repair_dsml(text.as_bytes()).map(|v| String::from_utf8(v).unwrap());
    let parse = |text: &str| {
        ds4_server::parse_generated_message(
            ModelSyntax::DeepSeek,
            text.as_bytes(),
            false,
            ChatFormat::DeepSeek,
            &[],
        )
    };

    // The wrapper close alone is missing: repaired, and the call parses.
    let full = format!(
        "thinking done\n\n{DSML_TOOL_CALLS_START}\n{DSML_INVOKE_START} name=\"bash\">\n\
         {DSML_PARAM_START} name=\"command\" string=\"true\">ls -la{DSML_PARAM_END}\n\
         {DSML_INVOKE_END}\n"
    );
    let repaired = repair(full.clone()).expect("a missing wrapper is the repairable case");
    assert_eq!(repaired, format!("{full}{DSML_TOOL_CALLS_END}"));
    let got = parse(&repaired);
    assert_eq!(got.calls.len(), 1);
    assert_eq!(got.calls[0].name, "bash");
    assert_eq!(got.calls[0].arguments, r#"{"command": "ls -la"}"#);

    // An unfinished invocation: refused.
    assert!(repair(format!(
        "\n\n{DSML_TOOL_CALLS_START}\n{DSML_INVOKE_START} name=\"edit\">\n\
         {DSML_PARAM_START} name=\"path\" string=\"true\">/tmp/test.c{DSML_PARAM_END}\n"
    ))
    .is_none());

    // An unfinished value: refused, even though the prefix looks plausible.
    assert!(repair(format!(
        "\n\n{DSML_TOOL_CALLS_START}\n{DSML_INVOKE_START} name=\"bash\">\n\
         {DSML_PARAM_START} name=\"command\" string=\"true\">echo hello"
    ))
    .is_none());

    // The older short spelling repairs the same way.
    let short = format!(
        "\n\n{DSML_TOOL_CALLS_START_SHORT}\n{DSML_INVOKE_START_SHORT} name=\"write_file\">\n\
         {DSML_PARAM_START_SHORT} name=\"path\" string=\"true\">/tmp/out.txt{DSML_PARAM_END_SHORT}\n\
         {DSML_PARAM_START_SHORT} name=\"content\" string=\"true\">hello world{DSML_PARAM_END_SHORT}\n\
         {DSML_INVOKE_END_SHORT}\n"
    );
    let repaired = repair(short.clone()).expect("the short spelling repairs");
    assert_eq!(repaired, format!("{short}{DSML_TOOL_CALLS_END_SHORT}"));
    let got = parse(&repaired);
    assert_eq!(got.calls.len(), 1);
    assert_eq!(got.calls[0].name, "write_file");
    assert_eq!(
        got.calls[0].arguments,
        r#"{"path": "/tmp/out.txt", "content": "hello world"}"#
    );

    // Plain XML: the same again.
    let plain = "\n\n<tool_calls>\n<invoke name=\"execute_command\">\n\
                 <parameter name=\"command\" string=\"true\">pwd</parameter>\n</invoke>\n"
        .to_string();
    let repaired = repair(plain).expect("the plain spelling repairs");
    let got = parse(&repaired);
    assert_eq!(got.calls.len(), 1);
    assert_eq!(got.calls[0].name, "execute_command");
    assert_eq!(got.calls[0].arguments, r#"{"command": "pwd"}"#);

    // Balanced text, no DSML, and a balanced block without an invocation: no
    // repair in any of them.
    assert!(repair(full.clone() + DSML_TOOL_CALLS_END).is_none());
    assert!(repair("just plain text, no tools".into()).is_none());
    let balanced_no_invoke = format!(
        "Let me analyze this.\n\n{DSML_TOOL_CALLS_START}The write tool truncates this too.\
         {DSML_TOOL_CALLS_END}"
    );
    assert!(repair(balanced_no_invoke).is_none());
    let balanced_short_no_invoke = format!(
        "thinking...\n\n{DSML_TOOL_CALLS_START_SHORT}some content here{DSML_TOOL_CALLS_END_SHORT}"
    );
    assert!(repair(balanced_short_no_invoke).is_none());
    let balanced_xml_no_invoke =
        "Let me think.\n\n<tool_calls>I need a tool but I don't know which.</tool_calls>"
            .to_string();
    assert!(repair(balanced_xml_no_invoke).is_none());

    // DSML quoted inside thinking is not executable, so the last </think>
    // scopes the scan and nothing follows it to repair.
    let thinking_quote = format!(
        "<think>The protocol uses {DSML_TOOL_CALLS_START}some text{DSML_TOOL_CALLS_END}, \
         but this is only a quote.</think>\nFinal answer."
    );
    assert!(repair(thinking_quote).is_none());

    // Extra closing tags are not a truncation pattern.
    let orphan =
        format!("done\n\n{DSML_TOOL_CALLS_START}{DSML_TOOL_CALLS_END}{DSML_TOOL_CALLS_END}");
    assert!(repair(orphan).is_none());

    // Quoted DSML before the last </think>, real DSML after it: repaired.
    let after_think = format!(
        "<think>{DSML_TOOL_CALLS_START}quoted, not executable{DSML_TOOL_CALLS_END}</think>\n\n\
         {DSML_TOOL_CALLS_START}\n{DSML_INVOKE_START} name=\"bash\">\n\
         {DSML_PARAM_START} name=\"command\" string=\"true\">date{DSML_PARAM_END}\n{DSML_INVOKE_END}\n"
    );
    let repaired = repair(after_think).expect("real DSML after thinking repairs");
    let got = parse(&repaired);
    assert_eq!(got.calls.len(), 1);
    assert_eq!(got.calls[0].name, "bash");
    assert_eq!(got.calls[0].arguments, r#"{"command": "date"}"#);
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
