//! C↔Rust four-surface JSON parsers (tokenize/render stay C).

use ds4_server::{
    generation_blocked, parse_anthropic_request, parse_chat_request, parse_completion_request,
    parse_request, parse_responses_request, ChatPart, ImageMime, ParseEnv, ParsedRequest,
    ToolChoice, WireSurface, NEED_IMAGE,
};

use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_PARSE_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/parse_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/parse_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_str(args: &[&str]) -> String {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run parse_c_oracle");
    assert!(
        out.status.success(),
        "oracle {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn env() -> ParseEnv {
    ParseEnv {
        default_model: "deepseek-v4-flash".into(),
        default_tokens: 393216,
        default_effort: ds4_server::ThinkMode::Low,
        default_temp: ds4_server::default_temperature(),
        live_ids: Vec::new(),
    }
}

fn rust_parse(surface: &str, body: &str) -> Result<ParsedRequest, String> {
    let e = env();
    match surface {
        "chat" => parse_chat_request(&e, body),
        "completion" => parse_completion_request(&e, body),
        "anthropic" => parse_anthropic_request(&e, body),
        "responses" => parse_responses_request(&e, body),
        _ => panic!("surface {surface}"),
    }
}

fn dump(r: &ParsedRequest) -> String {
    let mut s = format!(
        "OK\nkind={} api={} model={} from_req={}\nmax_tokens={} max_set={} top_k={} seed={}\ntemp={:.6} top_p={:.6} min_p={:.6}\nstream={} usage={} echo={} think={} tools={} tool_results={} choice={}\nstops={} summary={} needs={} nmsg={}\n",
        r.kind as i32,
        r.api as i32,
        r.model,
        r.model_from_request as i32,
        r.max_tokens,
        r.max_tokens_set as i32,
        r.top_k,
        r.seed,
        r.temperature,
        r.top_p,
        r.min_p,
        r.stream as i32,
        r.stream_include_usage as i32,
        r.return_token_ids as i32,
        r.think_mode as i32,
        r.has_tools as i32,
        r.has_tool_results as i32,
        r.tool_choice as i32,
        r.stops.len(),
        r.reasoning_summary_emit as i32,
        r.needs,
        r.messages.len()
    );
    for (i, m) in r.messages.iter().take(8).enumerate() {
        s.push_str(&format!(
            "msg{i}_role={} msg{i}_ncalls={}\n",
            m.role,
            m.calls.len()
        ));
    }
    s
}

fn both(surface: &str, body: &str) -> (Result<ParsedRequest, String>, String) {
    (rust_parse(surface, body), c_str(&[surface, body]))
}

fn normalize_c_schema_refusal(error: &str) -> String {
    error.replacen(
        "is not implemented: structured output is unsupported",
        "is not supported: structured output is unsupported",
        1,
    )
}

fn assert_err_eq(surface: &str, body: &str, expect: &str) {
    let (rust, c) = both(surface, body);
    let rust_e = rust.unwrap_err();
    assert!(c.starts_with("ERROR\n"), "{surface} {body}: {c}");
    let c_e = normalize_c_schema_refusal(c.strip_prefix("ERROR\n").unwrap().trim_end_matches('\n'));
    assert_eq!(rust_e, c_e, "{surface} {body}");
    assert!(
        rust_e.contains(expect) || rust_e == expect,
        "{surface} {body}: rust={rust_e:?} expect {expect:?}"
    );
}

fn assert_ok_eq(surface: &str, body: &str) -> ParsedRequest {
    let (rust, c) = both(surface, body);
    let rust = rust.unwrap_or_else(|e| panic!("rust {surface} {body}: {e}"));
    assert!(c.starts_with("OK\n"), "{surface} {body}: {c}");
    assert_eq!(dump(&rust), c, "{surface} {body}");
    rust
}

#[test]
fn c_unit_error_cases_match() {
    assert_err_eq("chat", r#"{"max_tokens":-1}"#, "max_tokens must be >= 0");
    assert_err_eq(
        "chat",
        r#"{"max_completion_tokens":-2}"#,
        "max_completion_tokens must be >= 0",
    );
    assert_err_eq(
        "anthropic",
        r#"{"max_tokens":-1}"#,
        "max_tokens must be >= 0",
    );
    assert_err_eq(
        "responses",
        r#"{"max_output_tokens":-3}"#,
        "max_output_tokens must be >= 0",
    );
    assert_err_eq(
        "completion",
        r#"{"max_tokens":-1}"#,
        "max_tokens must be >= 0",
    );

    assert_err_eq(
        "chat",
        r#"{"response_format":{"type":"text"}}"#,
        "missing messages",
    );
    assert_err_eq("chat", r#"{"response_format":null}"#, "missing messages");
    assert_err_eq("chat", r#"{"response_format":{}}"#, "missing messages");
    assert_err_eq(
        "chat",
        r#"{"response_format":{"type":"json_object"}}"#,
        "response_format type 'json_object' is not supported",
    );
    assert_err_eq(
        "chat",
        r#"{"response_format":{"json_schema":{"schema":{"type":"object"}},"type":"json_schema"}}"#,
        "'json_schema' is not supported",
    );
    assert_err_eq(
        "chat",
        r#"{"response_format":{"type":"xml"}}"#,
        "response_format type 'xml' is not supported",
    );
    assert_err_eq(
        "chat",
        r#"{"response_format":"json_object"}"#,
        "'json_object' is not supported",
    );
    assert_err_eq(
        "completion",
        r#"{"response_format":{"type":"text"}}"#,
        "missing prompt",
    );
    assert_err_eq(
        "completion",
        r#"{"response_format":{"type":"json_object"}}"#,
        "response_format type 'json_object' is not supported",
    );
    assert_err_eq(
        "responses",
        r#"{"text":{"format":{"type":"text"},"verbosity":"low"}}"#,
        "missing input",
    );
    assert_err_eq(
        "responses",
        r#"{"text":{"format":{"type":"json_schema","name":"x"}}}"#,
        "text.format type 'json_schema' is not supported",
    );
    assert_err_eq("anthropic", r#"{"output_format":null}"#, "missing messages");
    assert_err_eq(
        "anthropic",
        r#"{"output_format":{"type":"json_schema","schema":{}}}"#,
        "output_format type 'json_schema' is not supported",
    );
    assert_err_eq(
        "anthropic",
        r#"{"output_config":{"effort":"high","format":{"type":"json_object"}}}"#,
        "output_config.format type 'json_object' is not supported",
    );

    assert_err_eq(
        "responses",
        r#"{"previous_response_id":"resp_123"}"#,
        "previous_response_id is not supported; replay full input instead",
    );
    assert_err_eq(
        "responses",
        r#"{"conversation":"conv_1"}"#,
        "conversation is not supported; replay full input instead",
    );
}

#[test]
fn missing_required_fields_and_invalid_json() {
    assert_err_eq("chat", "{}", "missing messages");
    assert_err_eq("chat", "[]", "invalid JSON request");
    assert_err_eq("chat", "not-json", "invalid JSON request");
    assert_err_eq("completion", "{}", "missing prompt");
    assert_err_eq("anthropic", "{}", "missing messages");
    assert_err_eq("responses", "{}", "missing input");
    assert_err_eq(
        "chat",
        r#"{"messages":[{"role":"user","content":"hi"}]"#,
        "invalid JSON request",
    );
}

#[test]
fn tool_choice_errors_match_c() {
    let s = c_str(&["tool-choice-openai", r#""unsupported""#]);
    assert!(s.starts_with("ERROR\n"));
    assert!(s.contains("not supported"));
    assert_err_eq(
        "chat",
        r#"{"messages":[],"tool_choice":"unsupported"}"#,
        "tool_choice=unsupported not supported",
    );

    assert_err_eq(
        "chat",
        r#"{"messages":[],"tool_choice":{"type":"function"}}"#,
        "forced tool_choice not supported",
    );
    assert_err_eq(
        "anthropic",
        r#"{"messages":[],"tool_choice":{"type":"tool","name":"lookup"}}"#,
        "forced tool_choice not supported",
    );
    assert_err_eq(
        "chat",
        r#"{"messages":[],"tool_choice":"required"}"#,
        "tool_choice=required requires at least one tool",
    );
}

#[test]
fn tool_controls_are_not_silently_ignored() {
    let requests = [
        (
            "chat",
            r#"{"messages":[],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"tool_choice":"none"}"#,
        ),
        (
            "anthropic",
            r#"{"messages":[],"tools":[{"name":"lookup","input_schema":{"type":"object"}}],"tool_choice":{"type":"none"}}"#,
        ),
        (
            "responses",
            r#"{"input":[],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"tool_choice":"none"}"#,
        ),
    ];
    for (surface, body) in requests {
        let request = rust_parse(surface, body).unwrap();
        assert!(!request.has_tools, "{surface}");
        assert!(request.tool_schemas.is_empty(), "{surface}");
        assert!(request.tool_orders.is_empty(), "{surface}");
    }

    for surface in ["chat", "responses"] {
        let input = if surface == "chat" {
            r#"{"messages":[],"parallel_tool_calls":false}"#
        } else {
            r#"{"input":[],"parallel_tool_calls":false}"#
        };
        assert_eq!(
            rust_parse(surface, input).unwrap_err(),
            "parallel_tool_calls=false is not supported"
        );
    }
    assert!(rust_parse("chat", r#"{"messages":[],"parallel_tool_calls":true}"#).is_ok());
}

#[test]
fn simple_success_dumps_match_c() {
    let r = assert_ok_eq("chat", r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(r.think_mode as i32, 1);
    assert_eq!(r.needs, 6);

    let r = assert_ok_eq(
        "chat",
        r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"temperature":0.0,"thinking":{"type":"disabled"}}"#,
    );
    assert!(r.stream);
    assert_eq!(r.temperature, 0.0);
    assert_eq!(r.think_mode as i32, 0);

    assert_ok_eq("completion", r#"{"prompt":"hi","stream":true}"#);
    assert_ok_eq(
        "anthropic",
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#,
    );
    assert_ok_eq("responses", r#"{"input":"hi"}"#);
    assert_ok_eq("responses", r#"{"input":[]}"#);
    assert_ok_eq("chat", r#"{"messages":[]}"#);

    let r = assert_ok_eq(
        "chat",
        r#"{"messages":[{"role":"user","content":"hi"}],"model":"deepseek-chat"}"#,
    );
    assert_eq!(r.think_mode as i32, 0);

    let r = assert_ok_eq(
        "chat",
        r#"{"messages":[{"role":"user","content":"hi"}],"model":"deepseek-reasoner"}"#,
    );
    assert_eq!(r.think_mode as i32, 1);

    let r = assert_ok_eq(
        "anthropic",
        r#"{"system":"be brief","messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(r.messages.len(), 2);
    assert_eq!(r.messages[1].role, "system");

    let r = assert_ok_eq(
        "responses",
        r#"{"instructions":"sys","input":[{"type":"message","role":"user","content":"hi"}]}"#,
    );
    assert_eq!(r.messages[0].role, "system");
    assert_eq!(r.messages[1].role, "user");

    let r = assert_ok_eq(
        "chat",
        r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"tool_choice":"auto"}"#,
    );
    assert!(r.has_tools);
    assert_eq!(r.tool_choice, ToolChoice::Auto);
    assert_eq!(r.tool_orders.len(), 1);
    assert_eq!(r.tool_orders[0].name, "lookup");
}

#[test]
fn parse_request_dispatcher_and_needs() {
    let e = env();
    let r = parse_request(WireSurface::OpenaiChat, &e, r#"{"messages":[]}"#).unwrap();
    assert_eq!(r.kind as i32, 0);
    let r = parse_request(WireSurface::OpenaiCompletion, &e, r#"{"prompt":""}"#).unwrap();
    assert_eq!(r.kind as i32, 1);
}

#[test]
fn qwen_image_inputs_normalize_across_all_three_surfaces() {
    const PNG: &str = concat!(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8",
        "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    );
    let bodies = [
        format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"before"}},{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{PNG}"}}}},{{"type":"text","text":"after"}}]}}]}}"#
        ),
        format!(
            r#"{{"input":[{{"type":"message","role":"user","content":[{{"type":"input_text","text":"before"}},{{"type":"input_image","image_url":"data:image/png;base64,{PNG}"}},{{"type":"input_text","text":"after"}}]}}]}}"#
        ),
        format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"before"}},{{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{PNG}"}}}},{{"type":"text","text":"after"}}]}}]}}"#
        ),
    ];
    let parsed = [
        rust_parse("chat", &bodies[0]).unwrap(),
        rust_parse("responses", &bodies[1]).unwrap(),
        rust_parse("anthropic", &bodies[2]).unwrap(),
    ];
    for request in &parsed {
        assert_eq!(request.images.len(), 1);
        assert_eq!(request.images[0].mime, ImageMime::Png);
        assert_eq!(request.messages[0].content, "beforeafter");
        assert_eq!(
            request.messages[0].parts,
            [
                ChatPart::Text("before".into()),
                ChatPart::Image(0),
                ChatPart::Text("after".into()),
            ]
        );
        assert_ne!(request.needs & NEED_IMAGE, 0);
        assert_eq!(
            generation_blocked(request, 0),
            Some("image input is supported only by Qwen4Exp or GLM-5.3")
        );
        assert_eq!(
            generation_blocked(request, 6),
            Some("image input requires continuous runtime")
        );
        assert_eq!(generation_blocked(request, 7), None);
    }
    assert_eq!(parsed[0].images[0].data, parsed[1].images[0].data);
    assert_eq!(parsed[0].images[0].data, parsed[2].images[0].data);
}

#[test]
fn image_input_rejects_remote_urls_wrong_magic_and_non_user_roles() {
    let remote = r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}]}"#;
    assert_eq!(
        rust_parse("chat", remote).unwrap_err(),
        "image_url must be a base64 data URI"
    );

    const PNG: &str = concat!(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8",
        "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    );
    let wrong_magic = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":"data:image/jpeg;base64,{PNG}"}}]}}]}}"#
    );
    assert_eq!(
        rust_parse("chat", &wrong_magic).unwrap_err(),
        "image bytes do not match declared media type"
    );
    let assistant = format!(
        r#"{{"messages":[{{"role":"assistant","content":[{{"type":"image_url","image_url":"data:image/png;base64,{PNG}"}}]}}]}}"#
    );
    assert_eq!(
        rust_parse("chat", &assistant).unwrap_err(),
        "images are allowed only in user messages"
    );
}
