//! Corrective tool-call retry from `ds4_server.c` at v0.6.5-dfm:
//! tag-completion repair, model-visible tool-error suffix, and the
//! `decode_again` decision. Session rewind/sync stays on `DecodeIo`.

use crate::parse::ToolSchemaOrder;
use crate::render::{
    append_qwen_tool_response_text, append_solar_tool_response_text, append_tool_result_text,
    solar_role_open, ModelSyntax, QWEN_IM_END, QWEN_IM_START, QWEN_TOOL_RESPONSE_END,
    QWEN_TOOL_RESPONSE_START, SOLAR_IM_CONTENT, SOLAR_IM_END, SOLAR_IM_START, SOLAR_THINK_END,
    SOLAR_THINK_START, SOLAR_TOOL_ARG_END, SOLAR_TOOL_ARG_START, SOLAR_TOOL_ARG_VALUE,
    SOLAR_TOOL_CALLS, SOLAR_TOOL_CALL_END, SOLAR_TOOL_RESPONSE_END, SOLAR_TOOL_RESPONSE_START,
    THINK_HIGH_PREFIX, THINK_MAX_PREFIX,
};
use crate::route::{think_mode_enabled, ThinkMode};
use crate::stream::{think_end, ChatFormat};
use crate::tools::{
    parse_generated_message, DSML_INVOKE_END, DSML_INVOKE_END_SHORT, DSML_INVOKE_START,
    DSML_INVOKE_START_SHORT, DSML_PARAM_END, DSML_PARAM_END_SHORT, DSML_PARAM_START,
    DSML_PARAM_START_SHORT, DSML_TOOL_CALLS_END, DSML_TOOL_CALLS_END_SHORT, DSML_TOOL_CALLS_START,
    DSML_TOOL_CALLS_START_SHORT,
};

fn is_c_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn find_substr(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn find_last_substr(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len())
        .enumerate()
        .rev()
        .find(|(_, w)| *w == needle)
        .map(|(i, _)| i)
}

fn count_marker(s: &[u8], marker: &[u8]) -> usize {
    if marker.is_empty() {
        return 0;
    }
    let mut n = 0;
    let mut i = 0;
    while i + marker.len() <= s.len() {
        if s[i..].starts_with(marker) {
            n += 1;
            i += marker.len();
        } else {
            i += 1;
        }
    }
    n
}

pub fn syntax_skips_recovery(syntax: ModelSyntax) -> bool {
    matches!(
        syntax,
        ModelSyntax::Motif3 | ModelSyntax::Exaone | ModelSyntax::Dots3 | ModelSyntax::Glm53
    )
}

/// C `rendered_dsml_system_region`.
pub fn rendered_dsml_system_region(prompt: &[u8]) -> Vec<u8> {
    let bos = "<｜begin▁of▁sentence｜>".as_bytes();
    let mut p = if prompt.starts_with(bos) {
        &prompt[bos.len()..]
    } else {
        prompt
    };
    for prefix in [THINK_HIGH_PREFIX.as_bytes(), THINK_MAX_PREFIX.as_bytes()] {
        if !prefix.is_empty() && p.starts_with(prefix) {
            p = &p[prefix.len()..];
            break;
        }
    }
    while !p.is_empty() && is_c_space(p[0]) {
        p = &p[1..];
    }
    let user = find_substr(p, "<｜User｜>".as_bytes());
    let assistant = find_substr(p, "<｜Assistant｜>".as_bytes());
    let mut end = match (user, assistant) {
        (Some(u), Some(a)) => u.min(a),
        (Some(u), None) => u,
        (None, Some(a)) => a,
        (None, None) => p.len(),
    };
    while end > 0 && is_c_space(p[end - 1]) {
        end -= 1;
    }
    p[..end].to_vec()
}

/// C `rendered_solar_system_region`.
pub fn rendered_solar_system_region(prompt: &[u8]) -> Vec<u8> {
    let mut prefix = Vec::new();
    prefix.extend_from_slice(SOLAR_IM_START.as_bytes());
    prefix.extend_from_slice(b"system");
    prefix.extend_from_slice(SOLAR_IM_CONTENT.as_bytes());
    if !prompt.starts_with(&prefix) {
        return Vec::new();
    }
    let p = &prompt[prefix.len()..];
    let mut end = find_substr(p, SOLAR_IM_END.as_bytes()).unwrap_or(p.len());
    while end > 0 && is_c_space(p[end - 1]) {
        end -= 1;
    }
    p[..end].to_vec()
}

pub fn rendered_chat_system_region(format: ChatFormat, prompt: &[u8]) -> Vec<u8> {
    if format == ChatFormat::SolarOpen2 {
        rendered_solar_system_region(prompt)
    } else if format == ChatFormat::Qwen4Exp {
        let prefix = b"<|im_start|>system\n";
        let Some(mut body) = prompt.strip_prefix(prefix) else {
            return Vec::new();
        };
        if let Some(end) = find_substr(body, QWEN_IM_END.as_bytes()) {
            body = &body[..end];
        }
        body.trim_ascii().to_vec()
    } else {
        rendered_dsml_system_region(prompt)
    }
}

/// C `try_repair_dsml`.
pub fn try_repair_dsml(s: &[u8]) -> Option<Vec<u8>> {
    if s.is_empty() {
        return None;
    }
    let think = b"</think>";
    let scan = if let Some(i) = find_last_substr(s, think) {
        &s[i + think.len()..]
    } else {
        s
    };
    let (ts, te, is, ie, ps, pe) = if find_substr(scan, DSML_TOOL_CALLS_START.as_bytes()).is_some()
    {
        (
            DSML_TOOL_CALLS_START.as_bytes(),
            DSML_TOOL_CALLS_END.as_bytes(),
            DSML_INVOKE_START.as_bytes(),
            DSML_INVOKE_END.as_bytes(),
            DSML_PARAM_START.as_bytes(),
            DSML_PARAM_END.as_bytes(),
        )
    } else if find_substr(scan, DSML_TOOL_CALLS_START_SHORT.as_bytes()).is_some() {
        (
            DSML_TOOL_CALLS_START_SHORT.as_bytes(),
            DSML_TOOL_CALLS_END_SHORT.as_bytes(),
            DSML_INVOKE_START_SHORT.as_bytes(),
            DSML_INVOKE_END_SHORT.as_bytes(),
            DSML_PARAM_START_SHORT.as_bytes(),
            DSML_PARAM_END_SHORT.as_bytes(),
        )
    } else if find_substr(scan, b"<tool_calls>").is_some() {
        (
            b"<tool_calls>".as_slice(),
            b"</tool_calls>".as_slice(),
            b"<invoke".as_slice(),
            b"</invoke>".as_slice(),
            b"<parameter".as_slice(),
            b"</parameter>".as_slice(),
        )
    } else {
        return None;
    };
    let mut tos = 0usize;
    let mut toe = 0usize;
    let mut ios = 0usize;
    let mut ioe = 0usize;
    let mut pos = 0usize;
    let mut poe = 0usize;
    let mut i = 0;
    while i < scan.len() {
        if scan[i..].starts_with(ts) {
            tos += 1;
            i += ts.len();
        } else if scan[i..].starts_with(te) {
            toe += 1;
            i += te.len();
        } else if scan[i..].starts_with(is) {
            ios += 1;
            i += is.len();
        } else if scan[i..].starts_with(ie) {
            ioe += 1;
            i += ie.len();
        } else if scan[i..].starts_with(ps) {
            pos += 1;
            i += ps.len();
        } else if scan[i..].starts_with(pe) {
            poe += 1;
            i += pe.len();
        } else {
            i += 1;
        }
    }
    if tos == toe && ios == ioe && pos == poe {
        return None;
    }
    if toe > tos || ioe > ios || poe > pos {
        return None;
    }
    let mut out = s.to_vec();
    for _ in 0..(pos - poe) {
        out.extend_from_slice(pe);
    }
    for _ in 0..(ios - ioe) {
        out.extend_from_slice(ie);
    }
    for _ in 0..(tos - toe) {
        out.extend_from_slice(te);
    }
    Some(out)
}

/// C `try_repair_solar_tool_call`.
pub fn try_repair_solar(s: &[u8]) -> Option<Vec<u8>> {
    if s.is_empty() {
        return None;
    }
    let scan = if let Some(i) = find_last_substr(s, SOLAR_THINK_END.as_bytes()) {
        &s[i + SOLAR_THINK_END.len()..]
    } else {
        s
    };
    let call_open = count_marker(scan, SOLAR_TOOL_CALLS.as_bytes());
    let call_close = count_marker(scan, SOLAR_TOOL_CALL_END.as_bytes());
    let arg_open = count_marker(scan, SOLAR_TOOL_ARG_START.as_bytes());
    let arg_value = count_marker(scan, SOLAR_TOOL_ARG_VALUE.as_bytes());
    let arg_close = count_marker(scan, SOLAR_TOOL_ARG_END.as_bytes());
    if call_open == 0 || call_close > call_open || arg_close > arg_open || arg_value > arg_open {
        return None;
    }
    if call_open == call_close && arg_open == arg_close {
        return None;
    }
    if call_open != call_close + 1 || arg_open > arg_close + 1 {
        return None;
    }
    if arg_open != arg_value {
        return None;
    }
    let mut out = s.to_vec();
    if arg_open == arg_close + 1 {
        out.extend_from_slice(SOLAR_TOOL_ARG_END.as_bytes());
    }
    out.extend_from_slice(SOLAR_TOOL_CALL_END.as_bytes());
    Some(out)
}

pub fn try_repair_tool_call_format(format: ChatFormat, s: &[u8]) -> Option<Vec<u8>> {
    match format {
        ChatFormat::SolarOpen2 => try_repair_solar(s),
        ChatFormat::Exaone | ChatFormat::Qwen4Exp => None,
        ChatFormat::DeepSeek => try_repair_dsml(s),
    }
}

pub fn tool_call_format_has_repairable_truncation(format: ChatFormat, s: &[u8]) -> bool {
    try_repair_tool_call_format(format, s).is_some()
}

/// C `build_invalid_tool_error_suffix`.
pub fn build_invalid_tool_error_suffix(
    format: ChatFormat,
    think_mode: ThinkMode,
    thinking_inside: bool,
    prompt: &[u8],
    detail: &str,
) -> Vec<u8> {
    let solar = format == ChatFormat::SolarOpen2;
    let qwen = format == ChatFormat::Qwen4Exp;
    let system = rendered_chat_system_region(format, prompt);
    let mut tool_error = Vec::new();
    if solar {
        tool_error.extend_from_slice(b"Tool error: invalid Solar tool call");
    } else if qwen {
        tool_error.extend_from_slice(b"Tool error: invalid Qwen tool call");
    } else {
        tool_error.extend_from_slice(b"Tool error: invalid DSML tool call");
    }
    if !detail.is_empty() {
        tool_error.extend_from_slice(b": ");
        tool_error.extend_from_slice(detail.as_bytes());
    }
    if solar {
        tool_error.extend_from_slice(
            b"\nThe previous assistant output was not executed because the Solar tool syntax was malformed. \
Emit a new valid native Solar tool call, or answer normally if no tool is needed.",
        );
    } else if qwen {
        tool_error.extend_from_slice(
            b"\nThe previous assistant output was not executed because the Qwen tool syntax was malformed. \
Emit a new valid native Qwen tool call, or answer normally if no tool is needed.",
        );
    } else {
        tool_error.extend_from_slice(
            b"\nThe previous assistant output was not executed because the DSML syntax was malformed. \
Emit a new valid DSML tool call, or answer normally if no tool is needed.",
        );
    }
    if !system.is_empty() {
        tool_error.extend_from_slice(b"\n\nSystem prompt reminder:\n");
        tool_error.extend_from_slice(&system);
    }

    let mut suffix = Vec::new();
    if think_mode_enabled(think_mode) && thinking_inside {
        suffix.extend_from_slice(think_end(format).as_bytes());
    }
    if solar {
        suffix.extend_from_slice(SOLAR_IM_END.as_bytes());
        suffix.push(b'\n');
        solar_role_open(&mut suffix, "tool");
        suffix.extend_from_slice(SOLAR_TOOL_RESPONSE_START.as_bytes());
        append_solar_tool_response_text(&mut suffix, &tool_error);
        suffix.extend_from_slice(SOLAR_TOOL_RESPONSE_END.as_bytes());
        suffix.push(b'\n');
        suffix.extend_from_slice(SOLAR_IM_END.as_bytes());
        suffix.push(b'\n');
        solar_role_open(&mut suffix, "assistant");
        suffix.extend_from_slice(SOLAR_THINK_START.as_bytes());
        if !think_mode_enabled(think_mode) {
            suffix.extend_from_slice(SOLAR_THINK_END.as_bytes());
        }
    } else if qwen {
        suffix.extend_from_slice(QWEN_IM_END.as_bytes());
        suffix.extend_from_slice(b"\n");
        suffix.extend_from_slice(QWEN_IM_START.as_bytes());
        suffix.extend_from_slice(b"user\n");
        suffix.extend_from_slice(QWEN_TOOL_RESPONSE_START.as_bytes());
        suffix.extend_from_slice(b"\n");
        append_qwen_tool_response_text(&mut suffix, &tool_error);
        suffix.extend_from_slice(b"\n");
        suffix.extend_from_slice(QWEN_TOOL_RESPONSE_END.as_bytes());
        suffix.extend_from_slice(QWEN_IM_END.as_bytes());
        suffix.extend_from_slice(b"\n");
        suffix.extend_from_slice(QWEN_IM_START.as_bytes());
        suffix.extend_from_slice(b"assistant\n<think>\n");
        if !think_mode_enabled(think_mode) {
            suffix.extend_from_slice(b"\n</think>\n\n");
        }
    } else {
        suffix.extend_from_slice("<｜end▁of▁sentence｜><｜User｜><tool_result>".as_bytes());
        append_tool_result_text(&mut suffix, &tool_error);
        suffix.extend_from_slice("</tool_result><｜Assistant｜>".as_bytes());
        if think_mode_enabled(think_mode) {
            suffix.extend_from_slice(b"<think>");
        } else {
            suffix.extend_from_slice(b"</think>");
        }
    }
    suffix
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TruncationOutcome {
    None,
    Repair(Vec<u8>),
    RetryUnterminated,
    ErrorUnterminated,
}

/// C serial-loop truncation / repair / unterminated-recovery gate.
pub fn truncation_outcome(
    syntax: ModelSyntax,
    format: ChatFormat,
    kind_chat: bool,
    has_tools: bool,
    saw_tool_start: bool,
    saw_tool_end: bool,
    finish: &str,
    stream: bool,
    attempted: bool,
    text: &[u8],
    orders: &[ToolSchemaOrder],
) -> TruncationOutcome {
    if !kind_chat
        || !has_tools
        || syntax_skips_recovery(syntax)
        || !saw_tool_start
        || finish == "error"
    {
        return TruncationOutcome::None;
    }
    if saw_tool_end && !tool_call_format_has_repairable_truncation(format, text) {
        return TruncationOutcome::None;
    }
    if let Some(repaired) = try_repair_tool_call_format(format, text) {
        let parsed = parse_generated_message(syntax, &repaired, false, format, orders);
        if parsed.ok && !parsed.calls.is_empty() {
            return TruncationOutcome::Repair(repaired);
        }
    }
    if finish == "length" {
        return TruncationOutcome::None;
    }
    if !stream && !attempted {
        TruncationOutcome::RetryUnterminated
    } else {
        TruncationOutcome::ErrorUnterminated
    }
}

/// C parse-failure `decode_again` gate after `parse_generated_message_for_response`.
pub fn parse_failure_should_retry(
    syntax: ModelSyntax,
    stream: bool,
    attempted: bool,
    finish: &str,
    recovered: bool,
    has_tools: bool,
    saw_tool_start: bool,
) -> bool {
    recovered
        && has_tools
        && saw_tool_start
        && !syntax_skips_recovery(syntax)
        && !stream
        && !attempted
        && finish != "length"
}

pub fn terminal_finish(thinking_inside: bool, finish: &str) -> &'static str {
    if thinking_inside && finish != "error" && finish != "length" {
        "length"
    } else {
        match finish {
            "tool_calls" => "tool_calls",
            "length" => "length",
            "error" => "error",
            _ => "stop",
        }
    }
}

fn truncation_name(o: TruncationOutcome) -> &'static str {
    match o {
        TruncationOutcome::None => "none",
        TruncationOutcome::Repair(_) => "repair",
        TruncationOutcome::RetryUnterminated => "retry-unterminated",
        TruncationOutcome::ErrorUnterminated => "error-unterminated",
    }
}

fn utf(s: &[u8]) -> String {
    String::from_utf8_lossy(s).into_owned()
}

fn dsml_think_prompt() -> Vec<u8> {
    format!(
        "<｜begin▁of▁sentence｜>## Tools\nschema\n\nSystem rule\n\n<｜User｜>Hi<｜Assistant｜><think>"
    )
    .into_bytes()
}

fn solar_think_prompt() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(SOLAR_IM_START.as_bytes());
    p.extend_from_slice(b"system");
    p.extend_from_slice(SOLAR_IM_CONTENT.as_bytes());
    p.extend_from_slice(b"## System Prompt\n\nStay precise.");
    p.extend_from_slice(SOLAR_IM_END.as_bytes());
    p.push(b'\n');
    p.extend_from_slice(SOLAR_IM_START.as_bytes());
    p.extend_from_slice(b"user");
    p.extend_from_slice(SOLAR_IM_CONTENT.as_bytes());
    p.extend_from_slice(b"Look it up");
    p.extend_from_slice(SOLAR_IM_END.as_bytes());
    p.push(b'\n');
    p.extend_from_slice(SOLAR_IM_START.as_bytes());
    p.extend_from_slice(b"assistant");
    p.extend_from_slice(SOLAR_IM_CONTENT.as_bytes());
    p.extend_from_slice(SOLAR_THINK_START.as_bytes());
    p
}

fn repair_dsml_trunc() -> String {
    format!(
        "{}\n{} name=\"bash\">\n{} name=\"command\" string=\"true\">ls",
        DSML_TOOL_CALLS_START, DSML_INVOKE_START, DSML_PARAM_START
    )
}

fn repair_solar_trunc() -> String {
    format!(
        "Need a lookup.{SOLAR_THINK_END}{SOLAR_TOOL_CALLS}lookup\n{SOLAR_TOOL_ARG_START}query{SOLAR_TOOL_ARG_VALUE}solar open2"
    )
}

pub fn dump_script(name: &str) -> String {
    match name {
        "dsml-think" => utf(&build_invalid_tool_error_suffix(
            ChatFormat::DeepSeek,
            ThinkMode::Low,
            true,
            &dsml_think_prompt(),
            "missing invoke name",
        )),
        "dsml-nothink" => utf(&build_invalid_tool_error_suffix(
            ChatFormat::DeepSeek,
            ThinkMode::None,
            false,
            &dsml_think_prompt(),
            "invalid tool call",
        )),
        "solar-think" => utf(&build_invalid_tool_error_suffix(
            ChatFormat::SolarOpen2,
            ThinkMode::Low,
            true,
            &solar_think_prompt(),
            "missing argument terminator",
        )),
        "system-dsml" => utf(&rendered_dsml_system_region(&dsml_think_prompt())),
        "system-solar" => utf(&rendered_solar_system_region(&solar_think_prompt())),
        "repair-dsml" => match try_repair_dsml(repair_dsml_trunc().as_bytes()) {
            Some(v) => utf(&v),
            None => "NONE".into(),
        },
        "repair-solar" => match try_repair_solar(repair_solar_trunc().as_bytes()) {
            Some(v) => utf(&v),
            None => "NONE".into(),
        },
        "repair-dsml-none" => match try_repair_dsml(
            format!(
                "{}\n{} name=\"bash\">\n{} name=\"command\" string=\"true\">ls{}\n{}\n{}",
                DSML_TOOL_CALLS_START,
                DSML_INVOKE_START,
                DSML_PARAM_START,
                DSML_PARAM_END,
                DSML_INVOKE_END,
                DSML_TOOL_CALLS_END
            )
            .as_bytes(),
        ) {
            Some(v) => utf(&v),
            None => "NONE".into(),
        },
        "decide-unterminated-stop" => {
            let t = format!("{}\n{}>", DSML_TOOL_CALLS_START, DSML_INVOKE_START);
            format!(
                "{}\n",
                truncation_name(truncation_outcome(
                    ModelSyntax::DeepSeek,
                    ChatFormat::DeepSeek,
                    true,
                    true,
                    true,
                    false,
                    "stop",
                    false,
                    false,
                    t.as_bytes(),
                    &[],
                ))
            )
        }
        "decide-unterminated-length" => {
            let t = format!("{}\n{}>", DSML_TOOL_CALLS_START, DSML_INVOKE_START);
            format!(
                "{}\n",
                truncation_name(truncation_outcome(
                    ModelSyntax::DeepSeek,
                    ChatFormat::DeepSeek,
                    true,
                    true,
                    true,
                    false,
                    "length",
                    false,
                    false,
                    t.as_bytes(),
                    &[],
                ))
            )
        }
        "decide-parse-retry" => format!(
            "{}\n",
            parse_failure_should_retry(
                ModelSyntax::DeepSeek,
                false,
                false,
                "stop",
                true,
                true,
                true
            )
        ),
        "decide-parse-motif" => format!(
            "{}\n",
            parse_failure_should_retry(ModelSyntax::Motif3, false, false, "stop", true, true, true)
        ),
        _ => "ERROR unknown-script\n".into(),
    }
}
