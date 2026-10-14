//! C↔Rust four-surface JSON parsers (tokenize/render stay C).

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
        default_temp: ds4_server::default_temperature(),
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
            Some("image input is supported only by Qwen4Exp, GLM-5.3, Inkling, Step, Ling or MiMo")
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
        Some("image input is supported only by Qwen4Exp, GLM-5.3, Inkling, Step, Ling or MiMo")
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
        Some("video input requires MiMo")
    );
    assert_eq!(generation_blocked(&parsed, 12), None);
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
    // min_p defaults to the engine's 0.0 on this scope; the port's own default
    // (0.05) stays for the other families.
    assert_eq!(
        chat(r#"{"messages":[{"role":"user","content":"hi"}]}"#).min_p,
        0.0
    );
    assert_eq!(
        parse_chat_request(&env(), r#"{"messages":[{"role":"user","content":"hi"}]}"#)
            .unwrap()
            .min_p,
        0.05
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
