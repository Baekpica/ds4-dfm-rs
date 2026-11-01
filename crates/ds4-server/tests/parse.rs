//! C↔Rust four-surface JSON parsers (tokenize/render stay C).

use ds4_server::render::render_dsml_chat_choice;
use ds4_server::render::DSML_IMAGE;
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
        sampling: Default::default(),
        live_ids: Vec::new(),
        engine_defaults: false,
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

/// The C oracle keeps its old request defaults (seed 0, min_p 0.05); the
/// Rust host takes the model's and a fresh seed, so parity skips those two.
fn structural(dump: &str) -> String {
    dump.split(' ')
        .map(|f| {
            f.split('\n')
                .filter(|l| !l.starts_with("seed=") && !l.starts_with("min_p="))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn assert_ok_eq(surface: &str, body: &str) -> ParsedRequest {
    let (rust, c) = both(surface, body);
    let rust = rust.unwrap_or_else(|e| panic!("rust {surface} {body}: {e}"));
    assert!(c.starts_with("OK\n"), "{surface} {body}: {c}");
    assert_eq!(structural(&dump(&rust)), structural(&c), "{surface} {body}");
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

/// The accepted spelling of `response_format` is not a render input.  The
/// encoder oracle owns that rendering — it appends the schema to the SYSTEM
/// region as "## Response Format:\n\nYou MUST strictly adhere..." when a
/// message dict carries the field (encoding.py:64-66, :628-629) — and neither
/// reference server reads the field at all.  This server refuses the schema
/// forms by name (format.rs:9-15, the v0.6.3 refusal) and accepts "text"; the
/// case pins that the accepted form leaves every render input untouched, so a
/// later change cannot begin trusting a schema unnoticed.
#[test]
fn accepted_response_format_is_not_a_render_input() {
    let plain = rust_parse(
        "chat",
        r#"{"messages":[{"role":"user","content":"two words"}]}"#,
    )
    .unwrap();
    let text = rust_parse(
        "chat",
        r#"{"messages":[{"role":"user","content":"two words"}],"response_format":{"type":"text"}}"#,
    )
    .unwrap();

    assert_eq!(structural(&dump(&plain)), structural(&dump(&text)));
    assert_eq!(plain.tool_schemas, text.tool_schemas);
    assert!(plain.tool_schemas.is_empty());
    assert_eq!(plain.messages.len(), text.messages.len());
    assert_eq!(plain.messages[0].content, text.messages[0].content);
    assert_eq!(plain.messages[0].parts, text.messages[0].parts);
}

/// A chat tool's name reaches the prompt verbatim, and a tool-level `namespace`
/// key is not folded into it.  The encoder oracle does fold — `_tool_name_for_encoding`
/// (encoding.py:116) joins an explicit namespace or a "::"-qualified name into
/// "namespace::name", and :88-97 prefixes the namespace description onto the
/// function's — but that is the encoder's own rule for the message dicts handed
/// to it.  Neither reference server reads a tool-level namespace on the chat
/// surface: namespaces exist there for `type:"namespace"` items on the
/// Responses surface (server_parse_tools.c:9-120), which this port mirrors
/// (parse.rs:2147).  The case pins both forms this server does see.
#[test]
fn chat_tool_names_pass_through_verbatim() {
    // The qualified name the oracle would produce is taken as given.
    let qualified = r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"ns::foo","parameters":{"type":"object","properties":{}}}}]}"#;
    let parsed = rust_parse("chat", qualified).unwrap();
    assert_eq!(
        parsed.tool_schemas,
        r#"{"name":"ns::foo","parameters":{"type":"object","properties":{}}}"#
    );

    // The tool-level key is not part of the schema, and the function object is
    // carried byte-for-byte.
    let folded = r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","namespace":"ns","function":{"name":"foo","description":"d","parameters":{"type":"object","properties":{}}}}]}"#;
    let parsed = rust_parse("chat", folded).unwrap();
    assert_eq!(
        parsed.tool_schemas,
        r#"{"name":"foo","description":"d","parameters":{"type":"object","properties":{}}}"#
    );
}

#[test]
fn responses_namespace_tools_keep_both_spellings() {
    // The Responses surface is the one place a namespace exists
    // (server_parse_tools.c:9-120, mirrored by
    // `append_responses_namespace_tool_schemas`, parse.rs:2336).  A
    // `type:"namespace"` item holds its functions, the prompt advertises the
    // namespace concatenated with each function name
    // (`responses_namespace_function_schema`, parse.rs:2282), and the order
    // keeps both spellings so the answer can be emitted under the name the
    // client asked for (`responses_tool_wire_name`, stream.rs:234) plus the
    // namespace field (stream.rs:287).  The chat surface has no namespace at
    // all: `chat_tool_names_pass_through_verbatim` above pins that half.
    let body = r#"{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],"tools":[{"type":"namespace","name":"mcp__files","tools":[{"type":"function","name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}]}]}"#;
    let parsed = rust_parse("responses", body).unwrap();

    assert_eq!(
        parsed.tool_schemas,
        r#"{"name":"mcp__filesread","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}"#
    );
    assert_eq!(parsed.tool_orders.len(), 1);
    assert_eq!(parsed.tool_orders[0].name, "mcp__filesread");
    assert_eq!(parsed.tool_orders[0].wire_name, "read");
    assert_eq!(parsed.tool_orders[0].namespace, "mcp__files");

    // The prompt the model reads carries the same qualified name.
    let prompt = render_dsml_chat_choice(
        &parsed.messages,
        &parsed.tool_schemas,
        parsed.think_mode,
        parsed.tool_choice,
    );
    let prompt = String::from_utf8(prompt.unwrap()).unwrap();
    assert!(prompt.contains("mcp__filesread"), "{prompt}");
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

/// The named-function tool choice: the OpenAI chat reference defines
/// `{"type":"function","function":{"name":"my_function"}}` as "forces the model
/// to call that tool", and the port serves it by narrowing the request's tool
/// set to that function and requiring the call (OpenAI's own allowed_tools
/// mechanism for the same intent).  The production C engine refuses the form, so
/// the divergence is pinned below rather than left to a later parity pass to
/// find by accident.
#[test]
fn named_tool_choice_narrows_to_the_forced_function() {
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"other_function","parameters":{"type":"object"}}},{"type":"function","function":{"name":"my_function","parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}}],"tool_choice":{"type":"function","function":{"name":"my_function"}}}"#;
    let r = parse_chat_request(&env(), body).unwrap();
    assert_eq!(r.tool_choice, ToolChoice::Required);
    assert_eq!(r.forced_tool.as_deref(), Some("my_function"));
    assert!(r.has_tools);
    assert_eq!(r.tool_orders.len(), 1, "{:?}", r.tool_orders);
    assert_eq!(r.tool_orders[0].name, "my_function");
    assert!(r.tool_schemas.contains("my_function"), "{}", r.tool_schemas);
    assert!(
        !r.tool_schemas.contains("other_function"),
        "{}",
        r.tool_schemas
    );

    // Responses carries the same OpenAI value and takes the same route.
    let resp = parse_responses_request(
        &env(),
        r#"{"input":"hi","tools":[{"type":"function","function":{"name":"other_function","parameters":{"type":"object"}}},{"type":"function","function":{"name":"my_function","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"my_function"}}}"#,
    )
    .unwrap();
    assert_eq!(resp.tool_choice, ToolChoice::Required);
    assert_eq!(resp.tool_orders.len(), 1);
    assert!(resp.tool_schemas.contains("my_function"));
    assert!(!resp.tool_schemas.contains("other_function"));

    // The Responses wire form names the function at the top level, and a
    // pretty-printed tool list spans many lines: both narrow the same way.
    let resp = parse_responses_request(
        &env(),
        "{\"input\":\"hi\",\"tools\":[\n  {\n    \"type\": \"function\",\n    \"name\": \"other_function\",\n    \"parameters\": {\"type\": \"object\"}\n  },\n  {\n    \"type\": \"function\",\n    \"name\": \"my_function\",\n    \"parameters\": {\"type\": \"object\"}\n  }\n],\"tool_choice\":{\"type\":\"function\",\"name\":\"my_function\"}}",
    )
    .unwrap();
    assert_eq!(resp.tool_choice, ToolChoice::Required);
    assert_eq!(resp.tool_orders.len(), 1, "{:?}", resp.tool_orders);
    assert!(resp.has_tools);
    assert!(resp.tool_schemas.contains("my_function"));
    assert!(!resp.tool_schemas.contains("other_function"));
    let pretty = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[\n {\n  \"type\": \"function\",\n  \"function\": {\n   \"name\": \"my_function\",\n   \"parameters\": {\"type\": \"object\"}\n  }\n }\n],\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"my_function\"}}}";
    let pr = parse_chat_request(&env(), pretty).unwrap();
    assert!(
        pr.has_tools && pr.tool_schemas.contains("my_function"),
        "{}",
        pr.tool_schemas
    );

    // The renderer is handed exactly what the parse narrowed (generate.rs:1331),
    // so the served prompt offers the forced function alone and requires the
    // call.
    let prompt = render_dsml_chat_choice(&r.messages, &r.tool_schemas, r.think_mode, r.tool_choice);
    let prompt = String::from_utf8(prompt.unwrap()).unwrap();
    assert!(prompt.contains("my_function"), "{prompt}");
    assert!(!prompt.contains("other_function"), "{prompt}");
    assert!(prompt.contains("### Required Tool Use"), "{prompt}");

    // The C engine's refusal of the same body, recorded here.
    let c = c_str(&["chat", body]);
    assert!(c.starts_with("ERROR\n"), "{c}");
    assert!(c.contains("forced tool_choice not supported"), "{c}");
}

#[test]
fn named_tool_choice_rejects_a_name_outside_the_tools() {
    assert_eq!(
        parse_chat_request(
            &env(),
            r#"{"messages":[],"tools":[{"type":"function","function":{"name":"other_function","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"my_function"}}}"#,
        )
        .unwrap_err(),
        "tool_choice function \"my_function\" is not among the request tools"
    );
    assert_eq!(
        parse_chat_request(
            &env(),
            r#"{"messages":[],"tool_choice":{"type":"function","function":{"name":"my_function"}}}"#,
        )
        .unwrap_err(),
        "tool_choice function \"my_function\" is not among the request tools"
    );
}

#[test]
fn unsupported_tool_choice_objects_stay_refused() {
    let cases = [
        (
            r#"{"messages":[],"tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"auto","tools":[]}}}"#,
            "tool_choice type=allowed_tools not supported",
        ),
        (
            r#"{"messages":[],"tool_choice":{"type":"custom","custom":{"name":"my_function"}}}"#,
            "tool_choice type=custom not supported",
        ),
        (
            r#"{"messages":[],"tool_choice":{"type":"function"}}"#,
            "invalid tool_choice",
        ),
        (
            r#"{"messages":[],"tool_choice":{"type":"function","function":{}}}"#,
            "invalid tool_choice",
        ),
        (
            r#"{"messages":[],"tool_choice":{"function":{"name":"lookup"}}}"#,
            "invalid tool_choice",
        ),
        (r#"{"messages":[],"tool_choice":{}}"#, "invalid tool_choice"),
    ];
    for (body, expect) in cases {
        assert_eq!(
            parse_chat_request(&env(), body).unwrap_err(),
            expect,
            "{body}"
        );
    }
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
            Some("image input is supported only by Qwen4Exp, GLM-5.3, Inkling, Step, Ling, MiMo or MiniMax")
        );
        assert_eq!(
            generation_blocked(request, 6),
            Some("image input requires continuous runtime")
        );
        assert_eq!(generation_blocked(request, 7), None);
        assert_eq!(generation_blocked(request, 10), None);
        assert_eq!(generation_blocked(request, 11), None);
        assert_eq!(generation_blocked(request, 12), None);
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
    // An assistant image stays a parse error.  A tool/function image does not
    // (the MCP loop; see openai_tool_message_carries_the_image_placeholder).
    assert_eq!(
        rust_parse("chat", &assistant).unwrap_err(),
        "images are allowed only in user messages or tool results"
    );
}

/// The oracle's path and URL forms are refused on every surface.  Its
/// `parse_tagged_text` / `_extract_image` (encoding.py:273-320) normalize a bare
/// path or an http URL into an image record and leave the fetch to whatever
/// takes it; this server is base64-only by contract, and it fails closed rather
/// than dropping the pixels silently, so each form is a named request error.
#[test]
fn image_refusals_cover_the_oracle_path_and_url_forms() {
    // The oracle's <image>path</image> form, and the same path as a plain
    // string instead of the object spelling.
    for body in [
        r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"/tmp/photo.png"}}]}]}"#,
        r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":"/tmp/photo.png"}]}]}"#,
    ] {
        assert_eq!(
            rust_parse("chat", body).unwrap_err(),
            "image_url must be a base64 data URI"
        );
    }

    // Responses input_image with a URL; the data-URI form is pinned by
    // qwen_image_inputs_normalize_across_all_three_surfaces.
    let responses = r#"{"input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.com/x.png"}]}]}"#;
    assert_eq!(
        rust_parse("responses", responses).unwrap_err(),
        "image_url must be a base64 data URI"
    );

    // Anthropic's url source; its tool_result spelling of the same check is
    // pinned by anthropic_tool_result_content_blocks_carry_images.
    let anthropic = r#"{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}"#;
    assert_eq!(
        rust_parse("anthropic", anthropic).unwrap_err(),
        "Anthropic image source must use base64 data"
    );
}

#[test]
fn the_image_path_tag_form_is_text_and_not_markup() {
    // The oracle's `parse_tagged_text` rewrites an `<image>path</image>` text
    // form into `image_url` blocks and raises on an unbalanced tag
    // (encoding.py:273-297, pattern at :40).  This carriage serves no path form
    // at all -- pixels arrive base64 only (AGENTS.md, "Pixels arrive
    // base64-only") -- and it never reads the tag as markup, so the client's
    // text reaches the model byte for byte, as ordinary text.  Measured
    // 2026-10-20: no image is registered, the content keeps the tag, and the
    // rendered prompt carries it verbatim.
    for text in [
        "look at <image>/tmp/photo.png</image> please",
        "unbalanced <image> with no closer",
    ] {
        let body = format!(r#"{{"messages":[{{"role":"user","content":"{text}"}}]}}"#);
        let parsed = rust_parse("chat", &body).unwrap();

        assert!(parsed.images.is_empty(), "{text}: an image was registered");
        assert_eq!(
            parsed.needs & NEED_IMAGE,
            0,
            "{text}: an image span is owed"
        );
        assert_eq!(parsed.messages[0].content, text, "{text}: content moved");

        let prompt = render_dsml_chat_choice(
            &parsed.messages,
            &parsed.tool_schemas,
            parsed.think_mode,
            parsed.tool_choice,
        );
        let prompt = String::from_utf8(prompt.unwrap()).unwrap();
        assert!(prompt.contains(text), "{text}: the prompt rewrote the text");
    }
}

#[test]
fn anthropic_tool_result_content_blocks_carry_images() {
    // The engine's test_anthropic_tool_result_mcp_blocks
    // (tests/server_tests_replay.c:65): the real MCP loop replays an assistant
    // tool_use plus a user tool_result whose content is a block array, and a
    // screenshot-style MCP tool returns pixels inside it.
    const PNG: &str = concat!(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8",
        "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    );
    // The replaying assistant turn plus the tool_result block under test.
    let mcp_replay = |id: &str, block: &str| {
        format!(
            r#"{{"messages":[{{"role":"assistant","content":[{{"type":"tool_use","id":"{id}","name":"screenshot","input":{{}}}}]}},{{"role":"user","content":[{block}]}}]}}"#
        )
    };

    // Legacy string content renders byte-identically, including is_error:false.
    let string_form = rust_parse(
        "anthropic",
        &mcp_replay(
            "toolu_01",
            r#"{"type":"tool_result","tool_use_id":"toolu_01","is_error":false,"content":"ok"}"#,
        ),
    )
    .unwrap();
    assert_eq!(
        string_form.messages[1].content,
        "<tool_result>ok</tool_result>"
    );
    assert_eq!(
        string_form.messages[1].parts,
        [ChatPart::ToolResult {
            id: "toolu_01".into(),
            content: "ok".into(),
        }]
    );
    assert_eq!(string_form.messages[1].tool_call_ids, ["toolu_01"]);

    // Block array: text + image + text in client order.  The image registers
    // like a user-message image; the placeholder sits where the block sits in
    // the content the DSML head renders, while the ToolResult part keeps the
    // plain text the other heads read.
    let mixed = rust_parse(
        "anthropic",
        &mcp_replay(
            "toolu_02",
            &format!(
                r#"{{"type":"tool_result","tool_use_id":"toolu_02","content":[{{"type":"text","text":"shot:"}},{{"type":"image","source":{{"type":"base64","media_type":"image/png","data":"{PNG}"}}}},{{"type":"text","text":"done"}}]}}"#
            ),
        ),
    )
    .unwrap();
    assert_eq!(mixed.images.len(), 1);
    assert_eq!(mixed.images[0].mime, ImageMime::Png);
    assert_eq!(
        mixed.messages[1].content,
        format!("<tool_result>shot:{DSML_IMAGE}done</tool_result>")
    );
    assert_eq!(
        mixed.messages[1].parts,
        [
            ChatPart::Image(0),
            ChatPart::ToolResult {
                id: "toolu_02".into(),
                content: "shot:done".into(),
            },
        ]
    );
    assert_eq!(mixed.messages[1].tool_call_ids, ["toolu_02"]);
    assert_ne!(mixed.needs & NEED_IMAGE, 0);
    // The V4.1 head (variant 16) admits it.  A head that refuses images keeps
    // its own message; a head that admits them but cannot place one inside a
    // tool result (it renders the plain ToolResult part) is refused by name.
    assert_eq!(generation_blocked(&mixed, 16), None);
    assert_eq!(
        generation_blocked(&mixed, 0),
        Some("image input is supported only by Qwen4Exp, GLM-5.3, Inkling, Step, Ling, MiMo or MiniMax")
    );
    assert_eq!(
        generation_blocked(&mixed, 7),
        Some("images inside a tool result are supported only by DeepSeek V4.1 Flash")
    );

    // is_error:true surfaces as an explicit marker inside the envelope; a
    // text-only tool_result never needs pixels and renders as before.
    let errored = rust_parse(
        "anthropic",
        &mcp_replay(
            "toolu_03",
            r#"{"type":"tool_result","tool_use_id":"toolu_03","content":"exit status 1","is_error":true}"#,
        ),
    )
    .unwrap();
    assert_eq!(
        errored.messages[1].content,
        "<tool_result>[tool_error] exit status 1</tool_result>"
    );
    let text_only = rust_parse(
        "anthropic",
        &mcp_replay(
            "toolu_04",
            r#"{"type":"tool_result","tool_use_id":"toolu_04","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}"#,
        ),
    )
    .unwrap();
    assert_eq!(
        text_only.messages[1].content,
        "<tool_result>ab</tool_result>"
    );
    assert!(text_only.images.is_empty());

    // Fail closed: an image the server cannot deliver rejects the request
    // instead of silently vanishing.
    let url_source = mcp_replay(
        "toolu_05",
        r#"{"type":"tool_result","tool_use_id":"toolu_05","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}"#,
    );
    assert_eq!(
        rust_parse("anthropic", &url_source).unwrap_err(),
        "Anthropic image source must use base64 data"
    );
    let wrong_magic = mcp_replay(
        "toolu_06",
        &format!(
            r#"{{"type":"tool_result","tool_use_id":"toolu_06","content":[{{"type":"image","source":{{"type":"base64","media_type":"image/jpeg","data":"{PNG}"}}}}]}}"#
        ),
    );
    assert_eq!(
        rust_parse("anthropic", &wrong_magic).unwrap_err(),
        "image bytes do not match declared media type"
    );
}

#[test]
fn inkling_audio_chat_input() {
    const WAV: &str = "UklGRiYAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQIAAAAAAA==";
    let block =
        serde_json::json!({"type":"input_audio", "input_audio":{"format":"wav", "data":WAV}});
    let body = serde_json::json!({"messages":[{"role":"user", "content":[{"type":"text","text":"Transcribe."},block]}]});
    let result = rust_parse("chat", &body.to_string());
    assert!(result.is_ok(), "{result:?}");
    let parsed = result.unwrap();
    assert_eq!(parsed.audios.len(), 1);
    assert_eq!(
        parsed.messages[0].parts,
        [ChatPart::Text("Transcribe.".into()), ChatPart::Audio(0)]
    );
    assert_ne!(parsed.needs & ds4_server::route::NEED_AUDIO, 0);
    let route = ds4_server::route_decide(
        parsed.needs,
        WireSurface::OpenaiChat,
        &ds4_server::RouteEnv {
            have_cont: true,
            coalesce: true,
            prompt_len: 128,
            seq_cap: 1024,
            cont_anthropic: true,
            cont_responses: true,
            cont_tools_anthropic: true,
            cont_tools_responses: true,
        },
    );
    assert_eq!(route.lane, ds4_server::LANE_SERIAL);
    assert_eq!(generation_blocked(&parsed, 9), None);
    assert_eq!(
        generation_blocked(&parsed, 6),
        Some("audio input requires Inkling or MiMo")
    );
    let prompt = ds4_server::render_prompt(&parsed, 9).unwrap();
    assert!(String::from_utf8(prompt)
        .unwrap()
        .contains("<|content_audio_input|><|unused_200053|><|audio_end|>"));
    for bad in [
        body.to_string().replace("\"wav\"", "\"mp3\""),
        body.to_string().replace("\"user\"", "\"assistant\""),
        body.to_string().replace(WAV, "!!!!"),
        body.to_string().replace(WAV, "AAAA"),
        serde_json::json!({"messages":[{"role":"user","content":vec![block;5]}]}).to_string(),
    ] {
        assert!(rust_parse("chat", &bad).is_err(), "{bad}");
    }
}

#[test]
fn mimo_video_stays_on_the_serial_lane() {
    const MP4: &str = "AAAAAGZ0eXAAAAAAAAAAAA==";
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"video_url","video_url":{{"url":"data:video/mp4;base64,{MP4}"}}}}]}}]}}"#
    );
    let parsed = rust_parse("chat", &body).unwrap();
    assert_eq!(parsed.videos.len(), 1);
    assert_eq!(parsed.messages[0].parts, [ChatPart::Video(0)]);
    assert_ne!(parsed.needs & ds4_server::route::NEED_AUDIO, 0);
    let route = ds4_server::route_decide(
        parsed.needs,
        WireSurface::OpenaiChat,
        &ds4_server::RouteEnv {
            have_cont: true,
            coalesce: true,
            prompt_len: 128,
            seq_cap: 1024,
            cont_anthropic: true,
            cont_responses: true,
            cont_tools_anthropic: true,
            cont_tools_responses: true,
        },
    );
    assert_eq!(route.lane, ds4_server::LANE_SERIAL);
    assert_eq!(
        generation_blocked(&parsed, 6),
        Some("video input requires MiMo or MiniMax")
    );
    assert_eq!(generation_blocked(&parsed, 12), None);
    assert_eq!(generation_blocked(&parsed, 20), None);
    assert_eq!(
        rust_parse("chat", &body.replace("video/mp4", "video/webm")).unwrap_err(),
        "video input currently requires video/mp4"
    );
    assert_eq!(
        rust_parse("chat", &body.replace("\"user\"", "\"assistant\"")).unwrap_err(),
        "video is allowed only in user messages"
    );
}

#[test]
fn audio_image_budget_preflight() {
    // Two 10 MiB audio payloads exhaust the shared budget. A following
    // image must be refused before decoding or inspecting its bytes.
    let size = 10 * 1024 * 1024;
    let mut data = "UklGRgAAAABXQVZF".to_string();
    data.push_str(&"AAAA".repeat((size - 12) / 3));
    data.push_str("AA==");
    let block =
        serde_json::json!({"type":"input_audio", "input_audio":{"format":"wav", "data":data}});
    let body = serde_json::json!({"messages":[{"role":"user", "content":[block.clone(),block,{"type":"image_url","image_url":"data:image/png;base64,AAAA"}]}]});
    assert_eq!(
        rust_parse("chat", &body.to_string()).unwrap_err(),
        "media exceeds 20 MiB request limit"
    );
}

/// Unit H: the engine's request-omitted parse defaults and effort-name
/// collapse for its own model set (ParseEnv::engine_defaults, set by the
/// server bin only on the V4.1 route).  References: server_parse_chat.c:16-17
/// (thinking OFF, effort HIGH), server_msgs.c:194-215 (everything non-zero
/// below max collapses to HIGH), ds4.h:54-56 (temp 1.0 / top_p 1.0 / min_p
/// 0.0), server_parse_chat.c:91-127 (penalties + DRY).
#[test]
fn engine_defaults_think_and_sampling() {
    use ds4_server::ThinkMode;
    let eng = ParseEnv {
        engine_defaults: true,
        ..env()
    };
    let chat = |body: &str| parse_chat_request(&eng, body).unwrap();

    // Nothing sent: thinking OFF.
    assert_eq!(
        chat(r#"{"messages":[{"role":"user","content":"hi"}]}"#).think_mode,
        ThinkMode::None
    );
    // The engine's collapse: low / medium / minimal / xhigh / high -> HIGH.
    for name in ["low", "medium", "minimal", "xhigh", "high"] {
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":"hi"}}],"reasoning_effort":"{name}"}}"#
        );
        assert_eq!(chat(&body).think_mode, ThinkMode::High, "{name}");
    }
    assert_eq!(
        chat(r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"max"}"#)
            .think_mode,
        ThinkMode::Max
    );
    assert_eq!(
        chat(r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"}"#)
            .think_mode,
        ThinkMode::None
    );
    // A bare thinking flag: the engine's HIGH effort default.
    assert_eq!(
        chat(r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled"}}"#)
            .think_mode,
        ThinkMode::High
    );
    // The reasoner alias alone asks for thinking (server_msgs.c:304-306).
    assert_eq!(
        chat(r#"{"messages":[{"role":"user","content":"hi"}],"model":"deepseek-reasoner"}"#)
            .think_mode,
        ThinkMode::High
    );
    assert_eq!(
        chat(r#"{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#)
            .think_mode,
        ThinkMode::None
    );
    // min_p defaults to OpenAI's 0.0 unless the model or a flag sets it.
    assert_eq!(
        chat(r#"{"messages":[{"role":"user","content":"hi"}]}"#).min_p,
        0.0
    );
    assert_eq!(
        parse_chat_request(&env(), r#"{"messages":[{"role":"user","content":"hi"}]}"#)
            .unwrap()
            .min_p,
        0.0
    );
    // The penalties and DRY land on the request.
    let r = chat(
        r#"{"messages":[{"role":"user","content":"hi"}],"frequency_penalty":0.5,"presence_penalty":-0.2,"dry_multiplier":0.8}"#,
    );
    assert_eq!(r.frequency_penalty, 0.5);
    assert_eq!(r.presence_penalty, -0.2);
    assert!(r.dry_set);
    assert_eq!(r.dry_multiplier, 0.8);
}

/// O-01: the effort a request asked for survives the engine's collapse, so the
/// V4.1 head can render the requested budget (encoding.py:443-473) while the
/// engine keeps the mode it decided.  The name mapping is low 50 / high 75 /
/// max 100, and the oracle's numeric form renders as the number itself.
#[test]
fn requested_effort_budget_survives_the_engine_collapse() {
    use ds4_server::{RequestedEffort, ThinkMode};
    let eng = ParseEnv {
        engine_defaults: true,
        ..env()
    };
    let chat = |body: &str| parse_chat_request(&eng, body).unwrap();
    let ask = |effort: &str| {
        chat(&format!(
            r#"{{"messages":[{{"role":"user","content":"hi"}}],"reasoning_effort":{effort}}}"#
        ))
    };

    // The engine collapses low into HIGH (server_msgs.c:194-215); the head still
    // takes the low mapping's 50.
    let low = ask(r#""low""#);
    assert_eq!(low.think_mode, ThinkMode::High);
    assert_eq!(low.effort, RequestedEffort::named(ThinkMode::Low));
    assert_eq!(low.effort.head_budget(), Some(50));

    // A bare ask keeps the engine's HIGH default, 75 in the head.
    let bare = chat(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(bare.effort, RequestedEffort::named(ThinkMode::High));
    assert_eq!(bare.effort.head_budget(), Some(75));

    assert_eq!(ask(r#""max""#).effort.head_budget(), Some(100));
    assert_eq!(ask(r#""none""#).effort.head_budget(), None);

    // The oracle's numeric form: taken as the budget, engine mode collapsed.
    let numeric = ask("40");
    assert_eq!(numeric.think_mode, ThinkMode::High);
    assert_eq!(
        numeric.effort,
        RequestedEffort {
            mode: ThinkMode::High,
            budget: Some(40)
        }
    );
    assert_eq!(numeric.effort.head_budget(), Some(40));

    // Out of the oracle's range and its non-integers are refused, as the oracle
    // refuses them (encoding.py:464-469).
    for bad in ["0", "101", "50.5", "5e1"] {
        assert!(
            parse_chat_request(
                &eng,
                &format!(
                    r#"{{"messages":[{{"role":"user","content":"hi"}}],"reasoning_effort":{bad}}}"#
                )
            )
            .is_err(),
            "{bad} must be refused"
        );
    }
}

#[test]
fn openai_tool_message_carries_the_image_placeholder() {
    // The real MCP loop on the Chat Completions surface: an assistant turn
    // calls a screenshot tool, and the tool-role message's content is a block
    // array whose image block is the pixels.  The image registers like a
    // user-message image, and the placeholder lands in that message's content
    // in client block order, because that content is what the DSML render
    // writes inside its <tool_result> envelope.
    const PNG: &str = concat!(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8",
        "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    );
    let body = format!(
        r#"{{"messages":[{{"role":"assistant","content":"","tool_calls":[{{"id":"c1","type":"function","function":{{"name":"shot","arguments":"{{}}"}}}}]}},{{"role":"tool","tool_call_id":"c1","content":[{{"type":"text","text":"shot:"}},{{"type":"image_url","image_url":{{"url":"data:image/png;base64,{PNG}"}}}},{{"type":"text","text":"done"}}]}}]}}"#
    );
    let parsed = rust_parse("chat", &body).unwrap();
    assert_eq!(parsed.images.len(), 1);
    assert_eq!(parsed.images[0].mime, ImageMime::Png);
    assert_eq!(parsed.messages[1].role, "tool");
    assert_eq!(parsed.messages[1].content, format!("shot:{DSML_IMAGE}done"));
    assert_eq!(
        parsed.messages[1].parts,
        [
            ChatPart::Text("shot:".into()),
            ChatPart::Image(0),
            ChatPart::Text("done".into()),
        ]
    );
    assert_ne!(parsed.needs & NEED_IMAGE, 0);
    // The V4.1 head admits it; a head that admits images but cannot place one
    // inside a tool result is refused by name rather than leaking the token.
    assert_eq!(generation_blocked(&parsed, 16), None);
    assert_eq!(
        generation_blocked(&parsed, 7),
        Some("images inside a tool result are supported only by DeepSeek V4.1 Flash")
    );
}

mod v41_media {
    use super::*;
    use serde_json::{json, Value};

    const PNG: &str = concat!(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8",
        "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    );

    fn image(surface: &str, data: &str) -> Value {
        match surface {
            "chat" => {
                json!({"type":"image_url","image_url":format!("data:image/png;base64,{data}")})
            }
            "responses" => {
                json!({"type":"input_image","image_url":format!("data:image/png;base64,{data}")})
            }
            "anthropic" => {
                json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":data}})
            }
            _ => unreachable!(),
        }
    }

    fn request(surface: &str, blocks: Vec<Value>) -> Value {
        let messages = json!([{"role":"user","content":blocks}]);
        if surface == "responses" {
            json!({"input":messages})
        } else {
            json!({"messages":messages})
        }
    }

    fn parse(surface: &str, body: &Value) -> Result<ParsedRequest, String> {
        let mut env = env();
        env.engine_defaults = true;
        let body = body.to_string();
        match surface {
            "chat" => parse_chat_request(&env, &body),
            "anthropic" => parse_anthropic_request(&env, &body),
            "responses" => parse_responses_request(&env, &body),
            _ => unreachable!(),
        }
    }

    #[test]
    fn mixed_tool_image_order() {
        let nested = json!({"type":"tool_result","tool_use_id":"c1","content":[
            {"type":"text","text":"nested:"},image("anthropic", PNG),
            {"type":"text","text":":end"}
        ]});
        let outer = image("anthropic", PNG);
        for (blocks, want) in [
            (
                vec![outer.clone(), nested.clone()],
                format!("{DSML_IMAGE}<tool_result>nested:{DSML_IMAGE}:end</tool_result>"),
            ),
            (
                vec![nested, outer],
                format!("<tool_result>nested:{DSML_IMAGE}:end</tool_result>{DSML_IMAGE}"),
            ),
        ] {
            let body = json!({"messages":[
                {"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"shot","input":{}}]},
                {"role":"user","content":blocks}
            ]});
            let parsed = parse("anthropic", &body).unwrap();
            let rendered =
                String::from_utf8(ds4_server::render_prompt(&parsed, 16).unwrap()).unwrap();
            assert_eq!(parsed.images.len(), 2);
            assert!(rendered.contains(&want), "{rendered}");
            assert_eq!(rendered.matches(DSML_IMAGE).count(), 2);
            assert_eq!(
                parsed.messages[1]
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        ChatPart::Image(index) => Some(*index),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                [0, 1]
            );
        }
    }

    #[test]
    fn responses_tool_images() {
        for (call, output) in [
            ("function_call", "function_call_output"),
            ("custom_tool_call", "custom_tool_call_output"),
        ] {
            let body = json!({"reasoning":{"effort":"none"},"input":[
                {"type":call,"call_id":"c1","name":"shot","arguments":"{}"},
                {"type":output,"call_id":"c1","output":[
                    {"type":"input_text","text":"before:"},image("responses", PNG),
                    {"type":"input_text","text":":after"}
                ]}
            ]});
            let parsed = parse("responses", &body).unwrap();
            assert_eq!(parsed.images.len(), 1);
            assert_eq!(
                parsed.messages[1].content,
                format!("before:{DSML_IMAGE}:after")
            );
            assert_eq!(generation_blocked(&parsed, 16), None);
            assert!(generation_blocked(&parsed, 7)
                .unwrap()
                .contains("tool result"));
            let rendered =
                String::from_utf8(ds4_server::render_prompt(&parsed, 16).unwrap()).unwrap();
            assert!(rendered.contains(&format!(
                "<tool_result>before:{DSML_IMAGE}:after</tool_result>"
            )));
        }
    }

    #[test]
    fn multi_image_admission() {
        for surface in ["chat", "anthropic", "responses"] {
            for count in [5, 8] {
                let body = request(surface, vec![image(surface, PNG); count]);
                let parsed = parse(surface, &body).unwrap();
                assert_eq!(parsed.images.len(), count, "{surface}");
                assert!(rust_parse(surface, &body.to_string()).is_err(), "{surface}");
            }
        }
    }

    #[test]
    fn mixed_media_stays_bounded() {
        let audio =
            json!({"type":"input_audio","input_audio":{"format":"wav","data":"UklGRgAAAABXQVZF"}});
        for first in [true, false] {
            let mut blocks = vec![image("chat", PNG); 4];
            blocks.insert(if first { 0 } else { blocks.len() }, audio.clone());
            assert!(parse("chat", &request("chat", blocks)).is_err());
        }
    }

    #[test]
    fn image_bytes_stay_bounded() {
        let limit = 10 * 1024 * 1024;
        let mut data = "iVBORw0KGgoA".to_string();
        data.push_str(&"AAAA".repeat((limit - 9) / 3));
        data.push_str("AA==");
        let too_large = "AAAA".repeat(limit / 3 + 1);
        for surface in ["chat", "anthropic", "responses"] {
            let body = request(surface, vec![image(surface, &too_large)]);
            assert_eq!(
                parse(surface, &body).unwrap_err(),
                "media exceeds 10 MiB decoded limit"
            );
            let body = request(
                surface,
                vec![
                    image(surface, &data),
                    image(surface, &data),
                    image(surface, PNG),
                ],
            );
            assert_eq!(
                parse(surface, &body).unwrap_err(),
                "media exceeds 20 MiB request limit"
            );
        }
    }
}

/// CTR clause 2.3.7, the server's half: the client echoes `reasoning_content` on
/// every assistant message, empty included, and the server must not require the
/// key.  Both shapes parse, and the reasoning the message carried is what the
/// render sees -- an omitted key is not an error and an empty value stays empty
/// (`types.rs` 526-538; `conversation.rs` 3691-3740).
#[test]
fn assistant_reasoning_is_optional_and_may_be_empty() {
    let eng = ParseEnv {
        engine_defaults: true,
        ..env()
    };
    let empty = parse_chat_request(
        &eng,
        r#"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok","reasoning_content":""},{"role":"user","content":"again"}]}"#,
    )
    .unwrap();
    assert_eq!(empty.messages[1].role, "assistant");
    assert_eq!(empty.messages[1].content, "ok");
    assert_eq!(empty.messages[1].reasoning, "");

    let omitted = parse_chat_request(
        &eng,
        r#"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"},{"role":"user","content":"again"}]}"#,
    )
    .unwrap();
    assert_eq!(omitted.messages[1].content, "ok");
    assert_eq!(omitted.messages[1].reasoning, "");
}

/// A numeric reasoning_effort is the V4.1 head's budget; any other family
/// would read it as HIGH, so it is refused off the V4.1 route.
#[test]
fn numeric_effort_is_v41_only() {
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":5}"#;
    let v41 = ParseEnv {
        engine_defaults: true,
        ..env()
    };
    assert_eq!(
        parse_chat_request(&v41, body).unwrap().effort.budget,
        Some(5)
    );
    assert!(parse_chat_request(&env(), body).is_err());
}
