//! Endpoint-native error envelopes. Do not unify the families.

use crate::json::json_escape;
use crate::route::WireSurface;

pub fn http_reason(code: i32) -> &'static str {
    match code {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

pub fn openai_error_type(code: i32) -> &'static str {
    if code == 429 {
        "rate_limit_error"
    } else if code >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    }
}

pub fn anthropic_error_type(code: i32) -> &'static str {
    if code == 429 {
        "rate_limit_error"
    } else if code == 404 {
        "not_found_error"
    } else if code == 503 {
        "overloaded_error"
    } else if code >= 500 {
        "api_error"
    } else {
        "invalid_request_error"
    }
}

pub fn openai_error_body(code: i32, msg: &str) -> String {
    format!(
        "{{\"error\":{{\"message\":{},\"type\":\"{}\"}}}}\n",
        json_escape(msg),
        openai_error_type(code)
    )
}

pub fn anthropic_error_body(code: i32, msg: &str) -> String {
    format!(
        "{{\"type\":\"error\",\"error\":{{\"type\":\"{}\",\"message\":{}}}}}\n",
        anthropic_error_type(code),
        json_escape(msg)
    )
}

pub fn wire_error_body(surface: WireSurface, code: i32, msg: &str) -> String {
    if surface == WireSurface::Anthropic {
        anthropic_error_body(code, msg)
    } else {
        openai_error_body(code, msg)
    }
}

pub fn cors_headers() -> &'static str {
    "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n"
}

pub fn retry_after_header(sec: i32) -> String {
    format!("Retry-After: {sec}\r\n")
}

pub fn http_head(
    code: i32,
    content_type: Option<&str>,
    extra_headers: Option<&str>,
    cors: bool,
    body_len: usize,
) -> String {
    let mut h = format!(
        "HTTP/1.1 {code} {}\r\nContent-Length: {body_len}\r\n",
        http_reason(code)
    );
    if let Some(t) = content_type {
        if !t.is_empty() {
            h.push_str("Content-Type: ");
            h.push_str(t);
            h.push_str("\r\n");
        }
    }
    if let Some(e) = extra_headers {
        if !e.is_empty() {
            h.push_str(e);
        }
    }
    if cors {
        h.push_str(cors_headers());
    }
    h.push_str("Connection: close\r\n\r\n");
    h
}

pub fn http_response_bytes(
    code: i32,
    content_type: Option<&str>,
    extra_headers: Option<&str>,
    cors: bool,
    body: &str,
) -> Vec<u8> {
    let mut out = http_head(code, content_type, extra_headers, cors, body.len()).into_bytes();
    out.extend_from_slice(body.as_bytes());
    out
}

pub fn wire_http_error_bytes(
    surface: WireSurface,
    code: i32,
    msg: &str,
    cors: bool,
    retry_after: Option<i32>,
) -> Vec<u8> {
    let extra = retry_after.map(retry_after_header);
    let body = wire_error_body(surface, code, msg);
    http_response_bytes(
        code,
        Some("application/json"),
        extra.as_deref(),
        cors,
        &body,
    )
}

pub(crate) fn wire_stream_error_bytes(
    surface: WireSurface,
    msg: &str,
    responses_sequence: i32,
) -> Vec<u8> {
    let message = if msg.is_empty() {
        "internal server error"
    } else {
        msg
    };
    match surface {
        WireSurface::Anthropic => format!(
            "event: error\ndata: {{\"type\":\"error\",\"error\":{{\"type\":\"api_error\",\"message\":{}}}}}\n\n",
            json_escape(message)
        )
        .into_bytes(),
        WireSurface::Responses => format!(
            "data: {{\"type\":\"error\",\"sequence_number\":{responses_sequence},\"code\":\"server_error\",\"message\":{},\"param\":null}}\n\n",
            json_escape(message)
        )
        .into_bytes(),
        WireSurface::OpenaiChat | WireSurface::OpenaiCompletion => format!(
            "event: error\ndata: {{\"error\":{{\"message\":{},\"type\":\"server_error\"}}}}\n\n",
            json_escape(message)
        )
        .into_bytes(),
    }
}

#[cfg(test)]
mod stream_error_tests {
    use super::*;

    #[test]
    fn stream_errors_keep_each_surface_wire_shape() {
        assert_eq!(
            wire_stream_error_bytes(WireSurface::OpenaiChat, "boom", 0),
            b"event: error\ndata: {\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}\n\n"
        );
        assert_eq!(
            wire_stream_error_bytes(WireSurface::Anthropic, "boom", 0),
            b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"boom\"}}\n\n"
        );
        assert_eq!(
            wire_stream_error_bytes(WireSurface::Responses, "boom", 7),
            b"data: {\"type\":\"error\",\"sequence_number\":7,\"code\":\"server_error\",\"message\":\"boom\",\"param\":null}\n\n"
        );
    }
}
