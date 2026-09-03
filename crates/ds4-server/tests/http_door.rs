//! C↔Rust HTTP door: escape, envelopes, format refusal, model-id, /v1/models.

use ds4_server::{
    anthropic_error_body, anthropic_error_type, handle_client, header_accepts_json, header_chunked,
    header_end, http_head, json_escape, json_models_array_dup, model_alias_known,
    model_id_from_gguf_path, model_one_json, models_list_json, openai_error_body,
    openai_error_type, output_format_type_supported, parse_output_config_effort,
    parse_output_format_value, parse_responses_text_value, read_http_request, retry_after_header,
    wire_http_error_bytes, Json, ServerConfig, WireSurface,
};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::thread;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_SERVER_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/server_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/server_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_out(args: &[&str]) -> Vec<u8> {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run server_c_oracle");
    assert!(
        out.status.success(),
        "oracle {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn c_str(args: &[&str]) -> String {
    String::from_utf8(c_out(args)).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn json_escape_matches_c() {
    let rows = [
        "",
        "plain",
        "quote\"slash\\",
        "line\nret\rtab\t",
        "ctrl\u{0001}bell\u{0007}",
        "한글",
    ];
    for s in rows {
        assert_eq!(json_escape(s), c_str(&["escape", s]), "escape {s:?}");
    }
}

#[test]
fn error_envelopes_match_c() {
    let codes = [400, 404, 409, 429, 500, 503];
    for code in codes {
        assert_eq!(
            openai_error_type(code),
            c_str(&["openai-type", &code.to_string()]).trim()
        );
        assert_eq!(
            anthropic_error_type(code),
            c_str(&["anth-type", &code.to_string()]).trim()
        );
        let msg = "boom";
        assert_eq!(
            openai_error_body(code, msg),
            c_str(&["openai-error", &code.to_string(), msg])
        );
        assert_eq!(
            anthropic_error_body(code, msg),
            c_str(&["anth-error", &code.to_string(), msg])
        );
    }
    assert_eq!(
        openai_error_body(400, "say \"hi\"\n"),
        c_str(&["openai-error", "400", "say \"hi\"\n"])
    );
}

#[test]
fn http_head_and_wire_error_match_c() {
    let rust = http_head(404, Some("application/json"), None, false, 12);
    let c = c_str(&["http-head", "404", "application/json", "0", "-", "12"]);
    assert_eq!(rust, c);

    let rust = http_head(204, None, None, true, 0);
    let c = c_str(&["http-head", "204", "-", "1", "-", "0"]);
    assert_eq!(rust, c);
    assert!(rust.contains("Access-Control-Allow-Origin: *\r\n"));
    assert!(rust.contains("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"));
    assert!(!rust.contains("     Access-Control"));

    let extra = retry_after_header(5);
    let rust = http_head(429, Some("application/json"), Some(&extra), false, 3);
    let c = c_str(&[
        "http-head",
        "429",
        "application/json",
        "0",
        "Retry-After: 5\r\n",
        "3",
    ]);
    assert_eq!(rust, c);

    let rust = wire_http_error_bytes(
        WireSurface::OpenaiChat,
        400,
        "bad HTTP request",
        false,
        None,
    );
    let c = c_out(&["wire-http", "0", "400", "0", "-1", "bad HTTP request"]);
    assert_eq!(rust, c);

    let rust = wire_http_error_bytes(WireSurface::Anthropic, 503, "boom", true, Some(10));
    let c = c_out(&["wire-http", "2", "503", "1", "10", "boom"]);
    assert_eq!(rust, c);
    let s = String::from_utf8_lossy(&rust);
    assert!(s.contains("\"type\":\"overloaded_error\""));
    assert!(s.contains("Retry-After: 10\r\n"));
}

fn fmt_rust(field: &str, json: &str) -> Result<(), String> {
    let mut p = Json::new(json);
    parse_output_format_value(&mut p, field)
}

fn normalize_c_schema_refusal(error: &str) -> String {
    error.replacen(
        "is not implemented: structured output is unsupported",
        "is not supported: structured output is unsupported",
        1,
    )
}

fn fmt_c(field: &str, json: &str) -> Result<(), String> {
    let s = c_str(&["format-value", field, json]);
    if s.starts_with("OK\n") {
        Ok(())
    } else {
        Err(normalize_c_schema_refusal(
            s.strip_prefix("ERROR\n")
                .unwrap_or(&s)
                .trim_end_matches('\n'),
        ))
    }
}

#[test]
fn schema_format_refusal_matches_c() {
    let field = "response_format";
    assert!(output_format_type_supported(field, "text").is_ok());
    assert!(c_str(&["format-type", field, "text"]).starts_with("OK"));

    let refuse = output_format_type_supported(field, "json_object").unwrap_err();
    assert_eq!(refuse, fmt_c(field, "\"json_object\"").unwrap_err());
    assert!(refuse.contains("is not supported"));
    assert_eq!(
        output_format_type_supported(field, "xml").unwrap_err(),
        c_str(&["format-type", field, "xml"])
            .strip_prefix("ERROR\n")
            .unwrap()
            .trim_end_matches('\n')
    );

    let rows = [
        ("null", true),
        ("{}", true),
        ("{\"type\":\"text\"}", true),
        ("{\"type\":\"json_object\"}", false),
        (
            "{\"json_schema\":{\"schema\":{\"type\":\"object\"}},\"type\":\"json_schema\"}",
            false,
        ),
        ("{\"type\":\"xml\"}", false),
        ("\"json_object\"", false),
        ("\"text\"", true),
    ];
    for (json, ok) in rows {
        assert_eq!(fmt_rust(field, json).is_ok(), ok, "rust {json}");
        assert_eq!(fmt_c(field, json).is_ok(), ok, "c {json}");
        if !ok {
            assert_eq!(
                fmt_rust(field, json).unwrap_err(),
                fmt_c(field, json).unwrap_err(),
                "{json}"
            );
        }
    }

    let mut p =
        Json::new("{\"format\":{\"type\":\"json_schema\",\"name\":\"x\"},\"verbosity\":\"low\"}");
    let rust = parse_responses_text_value(&mut p).unwrap_err();
    let c = c_str(&[
        "text-value",
        "{\"format\":{\"type\":\"json_schema\",\"name\":\"x\"},\"verbosity\":\"low\"}",
    ]);
    assert!(c.starts_with("ERROR"));
    assert_eq!(
        rust,
        normalize_c_schema_refusal(c.strip_prefix("ERROR\n").unwrap().trim_end_matches('\n'))
    );
    assert!(rust.contains("text.format type 'json_schema' is not supported"));

    let mut p = Json::new("{\"effort\":\"high\",\"format\":{\"type\":\"json_object\"}}");
    let rust = parse_output_config_effort(&mut p).unwrap_err();
    let c = c_str(&[
        "output-config",
        "{\"effort\":\"high\",\"format\":{\"type\":\"json_object\"}}",
    ]);
    assert_eq!(
        rust,
        normalize_c_schema_refusal(c.strip_prefix("ERROR\n").unwrap().trim_end_matches('\n'))
    );

    let mut p = Json::new("{\"effort\":\"banana\",\"format\":{\"type\":\"text\"}}");
    assert!(parse_output_config_effort(&mut p).is_err());
    assert!(c_str(&[
        "output-config",
        "{\"effort\":\"banana\",\"format\":{\"type\":\"text\"}}"
    ])
    .starts_with("ERROR"));
}

#[test]
fn model_id_from_gguf_path_matches_c() {
    let paths = [
        "foo.gguf",
        "/tmp/foo.gguf",
        "/home/x/gguf/ds4flash.gguf",
        "/home/x/models/Motif-3-Mixed-Quant-GGUF/Motif-3-MQ87-88-FIT.gguf",
        "/home/x/models/Solar-Open2-250B-Mixed-Quant-GGUF/Solar-Open2-250B-MXQ-v1-00001-of-00011.gguf",
        "K-EXAONE-236B-A23B-MXQ-v1-00001-of-00003.gguf",
        "/tmp/",
        "",
    ];
    for path in paths {
        let rust = model_id_from_gguf_path(path);
        let c = c_str(&["model-id", path]);
        match rust {
            Some(id) => assert_eq!(format!("{id}\n"), c, "{path}"),
            None => assert_eq!(c, "NULL\n", "{path}"),
        }
    }
    assert!(model_alias_known("motif-3"));
    assert!(model_alias_known("Motif-3-Mixed-Quant-GGUF"));
    assert!(!model_alias_known("custom-id"));
}

#[test]
fn models_json_matches_c() {
    let id = "Motif-3-Mixed-Quant-GGUF";
    let rust = models_list_json(id, id, 196608, 1024, None);
    let c = c_str(&["models-list", id, id, "196608", "1024"]);
    assert_eq!(rust, c);
    assert!(rust.contains("\"owned_by\":\"ds4.c\""));
    assert!(rust.contains("\"created\":1767225600"));
    assert!(rust.ends_with('\n'));

    let rust = model_one_json(id, id, 8192, 393216);
    let c = c_str(&["model-one", id, id, "8192", "393216"]);
    assert_eq!(rust, c);

    let catalog = "{\"object\":\"list\",\"models\":[{\"id\":\"x\",\"context_window\":8192}]}";
    let rust = json_models_array_dup(catalog).unwrap();
    let c = c_str(&["models-array", catalog]);
    assert_eq!(rust, c);
    let rust_list = models_list_json("ds4", "ds4", 8192, 128, Some(&rust));
    let c_list = c_str(&["models-list", "ds4", "ds4", "8192", "128", &rust]);
    assert_eq!(rust_list, c_list);
}

#[test]
fn header_helpers_match_c() {
    let headers = [
        b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n".as_slice(),
        b"POST /v1/chat/completions HTTP/1.1\nContent-Length: 12abc\n\n".as_slice(),
        b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n".as_slice(),
        b"GET /x HTTP/1.1\r\nAccept: text/html, application/json\r\n\r\n".as_slice(),
        b"GET /x HTTP/1.1\nAccept: text/plain\n\n".as_slice(),
        b"POST /x HTTP/1.1\r\nContent-Length: -1\r\n\r\n".as_slice(),
    ];
    for h in headers {
        let hx = hex(h);
        assert_eq!(
            header_end(h).map(|n| n as i64).unwrap_or(-1),
            c_str(&["header-end", &hx]).trim().parse::<i64>().unwrap(),
            "{:?}",
            String::from_utf8_lossy(h)
        );
        assert_eq!(
            ds4_server::http::content_length(h),
            c_str(&["content-length", &hx])
                .trim()
                .parse::<i64>()
                .unwrap()
        );
        assert_eq!(
            header_chunked(h) as i32,
            c_str(&["header-chunked", &hx])
                .trim()
                .parse::<i32>()
                .unwrap()
        );
        assert_eq!(
            header_accepts_json(h) as i32,
            c_str(&["accept-json", &hx]).trim().parse::<i32>().unwrap()
        );
    }
}

#[test]
fn read_http_content_length_and_chunked() {
    let raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 5\r\n\r\nabcdeEXTRA";
    let req = read_http_request(&mut &raw[..], true).unwrap();
    assert_eq!(req.method, "POST");
    assert_eq!(req.path, "/v1/chat/completions");
    assert_eq!(req.body, b"abcde");

    let raw = b"GET /v1/models?foo=1 HTTP/1.1\r\n\r\n";
    let req = read_http_request(&mut &raw[..], true).unwrap();
    assert_eq!(req.path, "/v1/models");

    let raw =
        b"POST /v1/messages HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
    let req = read_http_request(&mut &raw[..], true).unwrap();
    assert_eq!(req.body, b"hello");

    let raw = b"POST /v1/messages HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 99\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
    let req = read_http_request(&mut &raw[..], true).unwrap();
    assert_eq!(req.body, b"hello");

    let raw =
        b"POST /v1/messages HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
    assert!(read_http_request(&mut &raw[..], false).is_some());
    let req = read_http_request(&mut &raw[..], false).unwrap();
    assert!(req.body.is_empty());

    let raw = b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nG\r\nxx\r\n0\r\n\r\n";
    assert!(read_http_request(&mut &raw[..], true).is_none());
}

#[test]
fn chunked_trailers_share_the_header_size_limit() {
    let mut raw = b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n".to_vec();
    for _ in 0..9 {
        raw.extend_from_slice(b"x:");
        raw.extend(std::iter::repeat_n(b'a', 8_180));
        raw.extend_from_slice(b"\r\n");
    }
    raw.extend_from_slice(b"\r\n");

    assert!(read_http_request(&mut &raw[..], true).is_none());
}

fn one_shot(cfg: &ServerConfig, req: &[u8]) -> Vec<u8> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = cfg.clone();
    let h = thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        handle_client(&cfg, &mut s);
    });
    let mut c = TcpStream::connect(addr).unwrap();
    c.write_all(req).unwrap();
    let _ = c.shutdown(std::net::Shutdown::Write);
    let mut out = Vec::new();
    c.read_to_end(&mut out).unwrap();
    h.join().unwrap();
    out
}

#[test]
fn tcp_models_options_unknown_bad_http() {
    let mut cfg = ServerConfig::default();
    cfg.model_id = "Motif-3-Mixed-Quant-GGUF".into();
    cfg.model_name = "Motif-3-Mixed-Quant-GGUF".into();
    cfg.ctx = 196608;
    cfg.default_tokens = 1024;
    cfg.cors = true;

    let out = one_shot(&cfg, b"GET /v1/models HTTP/1.1\r\n\r\n");
    let s = String::from_utf8_lossy(&out);
    assert!(s.starts_with("HTTP/1.1 200 OK\r\n"));
    assert!(s.contains("Access-Control-Allow-Origin: *"));
    let body = models_list_json(
        &cfg.model_id,
        &cfg.model_name,
        cfg.ctx,
        cfg.default_tokens,
        None,
    );
    assert!(s.ends_with(&body));

    let out = one_shot(&cfg, b"GET /v1/models/motif-3 HTTP/1.1\r\n\r\n");
    let s = String::from_utf8_lossy(&out);
    assert!(s.starts_with("HTTP/1.1 200 OK"));
    assert!(s.contains(&model_one_json(
        "motif-3",
        &cfg.model_name,
        cfg.ctx,
        cfg.default_tokens
    )));

    let out = one_shot(&cfg, b"OPTIONS /v1/chat/completions HTTP/1.1\r\n\r\n");
    let rust_head = http_head(204, None, None, true, 0);
    assert_eq!(out, rust_head.into_bytes());

    let out = one_shot(&cfg, b"GET /metrics HTTP/1.1\r\n\r\n");
    let s = String::from_utf8_lossy(&out);
    assert!(s.starts_with("HTTP/1.1 200 OK"));
    assert!(s.contains("ds4_uptime_seconds "));
    assert!(s.contains("ds4_route_requests_total{surface=\"openai_chat\",lane=\"serial\"} 0"));
    assert!(s.contains("ds4_requests_shed_total{reason=\"clients\"} 0"));
    assert!(s.contains("ds4_memory_census_supported 0"));
    assert!(s.contains("ds4_memory_substrate_outstanding_bytes 0"));

    let out = one_shot(&cfg, b"GET /v1/stats HTTP/1.1\r\n\r\n");
    let s = String::from_utf8_lossy(&out);
    assert!(s.starts_with("HTTP/1.1 200 OK"));
    assert!(s.contains("\"routes\":{"));
    assert!(s.contains("\"sheds\":{"));
    assert!(s.contains("\"memory\":{\"census_supported\":false"));
    assert!(s.contains("\"governor\":{\"shadow\":true"));

    let out = one_shot(
        &cfg,
        b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 45\r\n\r\n{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
    );
    assert_eq!(
        out,
        wire_http_error_bytes(
            WireSurface::OpenaiChat,
            503,
            "server shutting down",
            true,
            None,
        )
    );

    let out = one_shot(
        &cfg,
        b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}",
    );
    assert_eq!(
        out,
        wire_http_error_bytes(WireSurface::OpenaiChat, 400, "missing messages", true, None)
    );

    let out = one_shot(&cfg, b"NOTHTTP");
    assert_eq!(
        out,
        wire_http_error_bytes(WireSurface::OpenaiChat, 400, "bad HTTP request", true, None)
    );
}
