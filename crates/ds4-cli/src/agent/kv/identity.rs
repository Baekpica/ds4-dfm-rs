use ds4_kv::{sha1_hex, EXT_SESSION_TITLE};

fn le_put32(p: &mut [u8], v: u32) {
    p[0] = v as u8;
    p[1] = (v >> 8) as u8;
    p[2] = (v >> 16) as u8;
    p[3] = (v >> 24) as u8;
}

fn le_get32(p: &[u8]) -> u32 {
    u32::from(p[0]) | (u32::from(p[1]) << 8) | (u32::from(p[2]) << 16) | (u32::from(p[3]) << 24)
}

fn le_put64(p: &mut [u8], v: u64) {
    for i in 0..8 {
        p[i] = (v >> (8 * i)) as u8;
    }
}

const USER_MARK: &[u8] = "<｜User｜>".as_bytes();
const ASSISTANT_MARK: &[u8] = "<｜Assistant｜>".as_bytes();

#[derive(Clone, Debug, Default)]
pub(crate) struct SessionIdentity {
    pub(crate) title: Option<String>,
    pub(crate) created_at: u64,
    pub(crate) sha: Option<String>,
    pub(crate) has_user_turn: bool,
}

pub(crate) fn identity_sha(title: &str, created_at: u64) -> String {
    let mut bytes = Vec::with_capacity(title.len() + 8);
    bytes.extend_from_slice(title.as_bytes());
    let mut ts = [0u8; 8];
    le_put64(&mut ts, created_at);
    bytes.extend_from_slice(&ts);
    sha1_hex(&bytes)
}

pub(crate) fn text_sha(text: &[u8]) -> String {
    sha1_hex(text)
}

pub(crate) fn encode_title_trailer(title: &str) -> Vec<u8> {
    let mut out = vec![0u8; 4];
    le_put32(&mut out, title.len() as u32);
    out.extend_from_slice(title.as_bytes());
    out
}

pub(crate) fn decode_title_trailer(trailer: &[u8]) -> Option<String> {
    if trailer.len() < 4 {
        return None;
    }
    let n = le_get32(&trailer[..4]) as usize;
    trailer
        .get(4..4 + n)
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
}

pub(crate) fn file_identity_sha(
    ext_flags: u8,
    title: Option<&str>,
    created_at: u64,
    text: &[u8],
) -> String {
    if ext_flags & EXT_SESSION_TITLE != 0 {
        identity_sha(title.unwrap_or(""), created_at)
    } else {
        text_sha(text)
    }
}

pub(crate) fn title_from_prompt(prompt: &str) -> String {
    title_from_span(prompt.as_bytes(), 0, "(empty user prompt)")
}

pub(crate) fn title_from_text(text: &[u8], max_bytes: usize) -> String {
    let Some(start) = find_bytes(text, USER_MARK) else {
        return "(no user prompt)".into();
    };
    let body = &text[start + USER_MARK.len()..];
    let mut end = body.len();
    if let Some(at) = find_bytes(body, ASSISTANT_MARK) {
        end = end.min(at);
    }
    if let Some(at) = find_bytes(body, USER_MARK) {
        end = end.min(at);
    }
    title_from_span(&body[..end], max_bytes, "(empty user prompt)")
}

fn title_from_span(span: &[u8], max_bytes: usize, empty: &str) -> String {
    let limited = max_bytes != 0;
    let max_bytes = if limited { max_bytes.max(4) } else { 0 };
    let text = std::str::from_utf8(span).unwrap_or("");
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return empty.into();
    }
    let mut out = String::new();
    let mut pending_space = false;
    let mut truncated = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            pending_space = !out.is_empty();
            continue;
        }
        if pending_space && (!limited || out.len() + 4 < max_bytes) {
            out.push(' ');
            pending_space = false;
        }
        let next = ch.len_utf8();
        if limited && out.len() + next + 3 > max_bytes {
            truncated = true;
            break;
        }
        out.push(ch);
    }
    if truncated {
        out.push_str("...");
    }
    if out.is_empty() {
        empty.into()
    } else {
        out
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
