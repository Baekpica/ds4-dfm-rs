//! Blocking HTTP/1.1 reader. Content-Length + chunked (v0.6.3). No Tokio.

use std::io::{self, Read};

use crate::route::WireSurface;

pub const MAX_HEADER: usize = 64 * 1024;
pub const MAX_BODY: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
    pub accept_json: bool,
}

pub fn chunked_enabled() -> bool {
    std::env::var("DS4_SERVER_CHUNKED")
        .map(|v| v != "0")
        .unwrap_or(true)
}

pub fn header_end(p: &[u8]) -> Option<usize> {
    for i in 3..p.len() {
        if p[i - 3] == b'\r' && p[i - 2] == b'\n' && p[i - 1] == b'\r' && p[i] == b'\n' {
            return Some(i + 1);
        }
    }
    for i in 1..p.len() {
        if p[i - 1] == b'\n' && p[i] == b'\n' {
            return Some(i + 1);
        }
    }
    None
}

fn header_ci_eq(line: &[u8], name: &[u8]) -> bool {
    line.len() >= name.len() && line[..name.len()].eq_ignore_ascii_case(name)
}

pub fn header_accepts_json(h: &[u8]) -> bool {
    let mut i = 0;
    while i < h.len() {
        let start = i;
        while i < h.len() && h[i] != b'\n' {
            i += 1;
        }
        let mut len = i - start;
        if len > 0 && h[start + len - 1] == b'\r' {
            len -= 1;
        }
        let line = &h[start..start + len];
        if header_ci_eq(line, b"Accept:") {
            let mut j = 7;
            while j + 16 <= line.len() {
                if line[j..j + 16].eq_ignore_ascii_case(b"application/json") {
                    return true;
                }
                j += 1;
            }
        }
        if i < h.len() {
            i += 1;
        }
    }
    false
}

pub fn content_length(h: &[u8]) -> i64 {
    let mut i = 0;
    while i < h.len() {
        let start = i;
        while i < h.len() && h[i] != b'\n' {
            i += 1;
        }
        let mut len = i - start;
        if len > 0 && h[start + len - 1] == b'\r' {
            len -= 1;
        }
        let line = &h[start..start + len];
        if header_ci_eq(line, b"Content-Length:") {
            let mut v = &line[15..];
            while !v.is_empty() && v[0].is_ascii_whitespace() {
                v = &v[1..];
            }
            return parse_i64_c_strtol(v);
        }
        if i < h.len() {
            i += 1;
        }
    }
    0
}

pub fn header_chunked(h: &[u8]) -> bool {
    let mut i = 0;
    while i < h.len() {
        let start = i;
        while i < h.len() && h[i] != b'\n' {
            i += 1;
        }
        let mut len = i - start;
        if len > 0 && h[start + len - 1] == b'\r' {
            len -= 1;
        }
        let line = &h[start..start + len];
        if header_ci_eq(line, b"Transfer-Encoding:") {
            let mut j = 18;
            while j + 7 <= line.len() {
                if line[j..j + 7].eq_ignore_ascii_case(b"chunked") {
                    return true;
                }
                j += 1;
            }
        }
        if i < h.len() {
            i += 1;
        }
    }
    false
}

fn recv_more<R: Read>(r: &mut R, buf: &mut Vec<u8>, tmp_len: usize) -> io::Result<usize> {
    let mut tmp = vec![0u8; tmp_len];
    loop {
        match r.read(&mut tmp) {
            Ok(0) => return Ok(0),
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                return Ok(n);
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}

fn chunk_line_end<R: Read>(
    r: &mut R,
    buf: &mut Vec<u8>,
    pos: usize,
    max_line: usize,
) -> Option<usize> {
    let mut scan = pos;
    loop {
        while scan < buf.len() {
            if buf[scan] == b'\n' {
                return Some(scan + 1);
            }
            scan += 1;
        }
        if buf.len() - pos > max_line {
            return None;
        }
        match recv_more(r, buf, 8192) {
            Ok(0) | Err(_) => return None,
            Ok(_) => {}
        }
    }
}

fn read_chunked_body<R: Read>(
    r: &mut R,
    buf: &mut Vec<u8>,
    mut pos: usize,
    max_body: usize,
) -> Option<Vec<u8>> {
    let mut body = Vec::new();
    loop {
        if pos > 0 {
            buf.drain(..pos);
            pos = 0;
        }
        let le = chunk_line_end(r, buf, pos, 8192)?;
        if pos >= buf.len() || !buf[pos].is_ascii_hexdigit() {
            return None;
        }
        let line = &buf[pos..le];
        let mut endp = 0;
        while endp < line.len() && line[endp].is_ascii_hexdigit() {
            endp += 1;
        }
        let hex = std::str::from_utf8(&line[..endp]).ok()?;
        let sz = usize::from_str_radix(hex, 16).ok()?;
        let mut rest = &line[endp..];
        while !rest.is_empty() && (rest[0] == b' ' || rest[0] == b'\t') {
            rest = &rest[1..];
        }
        if rest.is_empty() || (rest[0] != b';' && rest[0] != b'\r' && rest[0] != b'\n') {
            return None;
        }
        pos = le;
        if sz == 0 {
            break;
        }
        if sz > max_body || body.len() + sz > max_body {
            return None;
        }
        while buf.len() < pos + sz {
            if recv_more(r, buf, 8192).ok()? == 0 {
                return None;
            }
        }
        body.extend_from_slice(&buf[pos..pos + sz]);
        pos += sz;
        let crlf = chunk_line_end(r, buf, pos, 2)?;
        let l2 = crlf - pos;
        if !(l2 == 1 || (l2 == 2 && buf[pos] == b'\r')) {
            return None;
        }
        pos = crlf;
    }
    let trailer_start = pos;
    loop {
        let le = chunk_line_end(r, buf, pos, 8192)?;
        if le - trailer_start > MAX_HEADER {
            return None;
        }
        let linelen = le - pos;
        let blank = linelen == 1 || (linelen == 2 && buf[pos] == b'\r');
        pos = le;
        if blank {
            break;
        }
    }
    Some(body)
}

pub fn read_http_request<R: Read>(r: &mut R, chunked_on: bool) -> Option<HttpRequest> {
    let mut b = Vec::new();
    let mut hend = None;
    while hend.is_none() && b.len() < MAX_HEADER {
        if recv_more(r, &mut b, 4096).ok()? == 0 {
            return None;
        }
        hend = header_end(&b);
    }
    let hend = hend?;
    let mut i = 0;
    let mut line = [0u8; 512];
    while i < b.len() && b[i] != b'\n' && i + 1 < line.len() {
        line[i] = b[i];
        i += 1;
    }
    let line = std::str::from_utf8(&line[..i]).ok()?;
    /* sscanf("%7s %255s"): stop at whitespace or the width. */
    let (method, rest) = scan_token(line, 7)?;
    let (path, _) = scan_token(rest, 255)?;
    let method = method.to_string();
    let mut path = path.to_string();
    if let Some(q) = path.find('?') {
        path.truncate(q);
    }
    let clen = content_length(&b[..hend]);
    if clen < 0 || clen as usize > MAX_BODY {
        return None;
    }
    let accept_json = header_accepts_json(&b[..hend]);
    if header_chunked(&b[..hend]) && chunked_on {
        let body = read_chunked_body(r, &mut b, hend, MAX_BODY)?;
        return Some(HttpRequest {
            method,
            path,
            body,
            accept_json,
        });
    }
    while b.len() < hend + clen as usize {
        if recv_more(r, &mut b, 8192).ok()? == 0 {
            return None;
        }
    }
    let body = b[hend..hend + clen as usize].to_vec();
    Some(HttpRequest {
        method,
        path,
        body,
        accept_json,
    })
}

pub fn shed_surface_for_path(path: &str) -> Option<WireSurface> {
    match path {
        "/v1/messages" => Some(WireSurface::Anthropic),
        "/v1/chat/completions" => Some(WireSurface::OpenaiChat),
        "/v1/responses" => Some(WireSurface::Responses),
        "/v1/completions" => Some(WireSurface::OpenaiCompletion),
        _ => None,
    }
}

pub fn parse_surface_for_path(path: &str) -> Option<WireSurface> {
    shed_surface_for_path(path)
}

fn parse_i64_c_strtol(v: &[u8]) -> i64 {
    let mut i = 0;
    while i < v.len() && matches!(v[i], b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
        i += 1;
    }
    if i >= v.len() {
        return 0;
    }
    let mut neg = false;
    if v[i] == b'+' {
        i += 1;
    } else if v[i] == b'-' {
        neg = true;
        i += 1;
    }
    if i >= v.len() || !v[i].is_ascii_digit() {
        return 0;
    }
    let mut n: i64 = 0;
    while i < v.len() && v[i].is_ascii_digit() {
        n = n.saturating_mul(10).saturating_add((v[i] - b'0') as i64);
        i += 1;
    }
    if neg {
        -n
    } else {
        n
    }
}

fn scan_token(s: &str, max: usize) -> Option<(&str, &str)> {
    let s = s.trim_start_matches(|c: char| c == ' ' || c == '\t');
    if s.is_empty() {
        return None;
    }
    let n = s
        .find(|c: char| c == ' ' || c == '\t' || c == '\r')
        .unwrap_or(s.len())
        .min(max);
    if n == 0 {
        return None;
    }
    Some((&s[..n], &s[n..]))
}
