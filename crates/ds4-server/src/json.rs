//! Request JSON walker copied from `ds4_server.c`. Ceiling 256 on skip nest.

pub const JSON_MAX_NESTING: i32 = 256;

#[derive(Debug)]
pub struct Json<'a> {
    s: &'a [u8],
    pub i: usize,
}

impl<'a> Json<'a> {
    pub fn new(s: &'a str) -> Self {
        Self {
            s: s.as_bytes(),
            i: 0,
        }
    }

    pub fn from_bytes(s: &'a [u8]) -> Self {
        Self { s, i: 0 }
    }

    pub fn remaining(&self) -> &[u8] {
        &self.s[self.i..]
    }

    pub fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    pub fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.i += 1;
        Some(c)
    }

    pub fn ws(&mut self) {
        while matches!(self.peek(), Some(c) if is_c_space(c)) {
            self.i += 1;
        }
    }

    pub fn lit(&mut self, lit: &str) -> bool {
        let b = lit.as_bytes();
        if self.s.get(self.i..self.i + b.len()) == Some(b) {
            self.i += b.len();
            true
        } else {
            false
        }
    }
}

fn is_c_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn hex(c: u8) -> Option<u32> {
    match c {
        b'0'..=b'9' => Some((c - b'0') as u32),
        b'a'..=b'f' => Some(10 + (c - b'a') as u32),
        b'A'..=b'F' => Some(10 + (c - b'A') as u32),
        _ => None,
    }
}

fn utf8_put(out: &mut Vec<u8>, cp: u32) {
    if cp <= 0x7f {
        out.push(cp as u8);
    } else if cp <= 0x7ff {
        out.push((0xc0 | (cp >> 6)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else if cp <= 0xffff {
        out.push((0xe0 | (cp >> 12)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else {
        out.push((0xf0 | (cp >> 18)) as u8);
        out.push((0x80 | ((cp >> 12) & 0x3f)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    }
}

fn json_u16(p: &mut Json<'_>) -> Option<u32> {
    if p.peek() != Some(b'\\') {
        return None;
    }
    if p.s.get(p.i + 1) != Some(&b'u') {
        return None;
    }
    let mut cp = 0u32;
    for k in 0..4 {
        let h = hex(*p.s.get(p.i + 2 + k)?)?;
        cp = (cp << 4) | h;
    }
    p.i += 6;
    Some(cp)
}

pub fn json_string(p: &mut Json<'_>) -> Option<String> {
    p.ws();
    if p.bump() != Some(b'"') {
        return None;
    }
    let mut b = Vec::new();
    while let Some(c) = p.peek() {
        if c == b'"' {
            break;
        }
        p.i += 1;
        if c != b'\\' {
            b.push(c);
            continue;
        }
        let e = p.bump()?;
        match e {
            b'"' | b'\\' | b'/' => b.push(e),
            b'b' => b.push(b'\x08'),
            b'f' => b.push(b'\x0c'),
            b'n' => b.push(b'\n'),
            b'r' => b.push(b'\r'),
            b't' => b.push(b'\t'),
            b'u' => {
                p.i -= 2;
                let mut cp = json_u16(p)?;
                if (0xd800..=0xdbff).contains(&cp) {
                    let save = p.i;
                    if let Some(lo) = json_u16(p) {
                        if (0xdc00..=0xdfff).contains(&lo) {
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                        } else {
                            p.i = save;
                        }
                    }
                }
                utf8_put(&mut b, cp);
            }
            _ => return None,
        }
    }
    if p.bump() != Some(b'"') {
        return None;
    }
    String::from_utf8(b).ok()
}

pub fn json_number(p: &mut Json<'_>) -> Option<f64> {
    p.ws();
    let start = p.i;
    if matches!(p.peek(), Some(b'+' | b'-')) {
        p.i += 1;
    }
    let after_sign = p.i;
    while matches!(p.peek(), Some(c) if c.is_ascii_digit()) {
        p.i += 1;
    }
    if p.peek() == Some(b'.') {
        p.i += 1;
        while matches!(p.peek(), Some(c) if c.is_ascii_digit()) {
            p.i += 1;
        }
    }
    if matches!(p.peek(), Some(b'e' | b'E')) {
        p.i += 1;
        if matches!(p.peek(), Some(b'+' | b'-')) {
            p.i += 1;
        }
        while matches!(p.peek(), Some(c) if c.is_ascii_digit()) {
            p.i += 1;
        }
    }
    if p.i == after_sign {
        p.i = start;
        return None;
    }
    let s = std::str::from_utf8(&p.s[start..p.i]).ok()?;
    s.parse().ok()
}

pub fn json_int(p: &mut Json<'_>) -> Option<i32> {
    let mut v = json_number(p)?;
    if v < 0.0 {
        v = 0.0;
    }
    if v > i32::MAX as f64 {
        v = i32::MAX as f64;
    }
    Some(v as i32)
}

pub fn json_bool(p: &mut Json<'_>) -> Option<bool> {
    p.ws();
    if p.lit("true") {
        return Some(true);
    }
    if p.lit("false") {
        return Some(false);
    }
    None
}

fn skip_value_depth(p: &mut Json<'_>, depth: i32) -> bool {
    p.ws();
    match p.peek() {
        Some(b'"') => json_string(p).is_some(),
        Some(b'{') => skip_object_depth(p, depth),
        Some(b'[') => skip_array_depth(p, depth),
        Some(_) if p.lit("true") || p.lit("false") || p.lit("null") => true,
        Some(_) => json_number(p).is_some(),
        None => false,
    }
}

fn skip_array_depth(p: &mut Json<'_>, depth: i32) -> bool {
    if depth >= JSON_MAX_NESTING {
        return false;
    }
    p.ws();
    if p.bump() != Some(b'[') {
        return false;
    }
    p.ws();
    if p.peek() == Some(b']') {
        p.i += 1;
        return true;
    }
    loop {
        if !skip_value_depth(p, depth + 1) {
            return false;
        }
        p.ws();
        if p.peek() == Some(b']') {
            p.i += 1;
            return true;
        }
        if p.bump() != Some(b',') {
            return false;
        }
    }
}

fn skip_object_depth(p: &mut Json<'_>, depth: i32) -> bool {
    if depth >= JSON_MAX_NESTING {
        return false;
    }
    p.ws();
    if p.bump() != Some(b'{') {
        return false;
    }
    p.ws();
    if p.peek() == Some(b'}') {
        p.i += 1;
        return true;
    }
    loop {
        if json_string(p).is_none() {
            return false;
        }
        p.ws();
        if p.bump() != Some(b':') {
            return false;
        }
        if !skip_value_depth(p, depth + 1) {
            return false;
        }
        p.ws();
        if p.peek() == Some(b'}') {
            p.i += 1;
            return true;
        }
        if p.bump() != Some(b',') {
            return false;
        }
    }
}

pub fn json_skip_value(p: &mut Json<'_>) -> bool {
    skip_value_depth(p, 0)
}

pub fn json_raw_value(p: &mut Json<'_>) -> Option<String> {
    p.ws();
    let start = p.i;
    if !json_skip_value(p) {
        return None;
    }
    String::from_utf8(p.s[start..p.i].to_vec()).ok()
}

pub fn json_content(p: &mut Json<'_>) -> Option<String> {
    p.ws();
    if p.peek() == Some(b'"') {
        return json_string(p);
    }
    if p.lit("null") {
        return Some(String::new());
    }
    if p.peek() != Some(b'[') {
        if !json_skip_value(p) {
            return None;
        }
        return Some(String::new());
    }
    p.i += 1;
    let mut b = String::new();
    p.ws();
    while p.peek().is_some() && p.peek() != Some(b']') {
        if p.peek() == Some(b'"') {
            b.push_str(&json_string(p)?);
        } else if p.peek() == Some(b'{') {
            p.i += 1;
            p.ws();
            while p.peek().is_some() && p.peek() != Some(b'}') {
                let key = json_string(p)?;
                p.ws();
                if p.bump() != Some(b':') {
                    return None;
                }
                if key == "text" {
                    b.push_str(&json_string(p)?);
                } else if !json_skip_value(p) {
                    return None;
                }
                p.ws();
                if p.peek() == Some(b',') {
                    p.i += 1;
                }
                p.ws();
            }
            if p.bump() != Some(b'}') {
                return None;
            }
        } else if !json_skip_value(p) {
            return None;
        }
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.bump() != Some(b']') {
        return None;
    }
    Some(b)
}

/// Surrounding quotes included, matching C `json_escape`. Walks raw bytes
/// so a held-back UTF-8 fragment can be escaped the same way as C.
pub fn json_escape_bytes(s: &[u8]) -> Vec<u8> {
    let mut out = vec![b'"'];
    for &c in s {
        match c {
            b'"' | b'\\' => {
                out.push(b'\\');
                out.push(c);
            }
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            c if c < 0x20 => {
                let esc = format!("\\u{c:04x}");
                out.extend_from_slice(esc.as_bytes());
            }
            c => out.push(c),
        }
    }
    out.push(b'"');
    out
}

/// Surrounding quotes included, matching `json_escape`.
pub fn json_escape(s: &str) -> String {
    String::from_utf8(json_escape_bytes(s.as_bytes())).expect("json_escape preserves UTF-8")
}

/// C `json_minify_raw_value`: drop insignificant whitespace outside strings.
pub fn json_minify_raw_value(json: &str) -> String {
    let src = if json.is_empty() { "null" } else { json };
    let mut p = Json::new(src);
    p.ws();
    let start = p.i;
    if !json_skip_value(&mut p) {
        return src.to_string();
    }
    let end = p.i;
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    for &c in &bytes[start..end] {
        if in_string {
            out.push(c);
            if escape {
                escape = false;
            } else if c == b'\\' {
                escape = true;
            } else if c == b'"' {
                in_string = false;
            }
        } else if c == b'"' {
            in_string = true;
            out.push(c);
        } else if !is_c_space(c) {
            out.push(c);
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| src.to_string())
}

#[derive(Debug, Clone)]
pub struct JsonArg {
    pub key: String,
    pub value: String,
    pub is_string: bool,
    pub used: bool,
}

/// C `json_args_parse`. Does not consume the closing `}`.
pub fn json_args_parse(json: &str) -> Option<Vec<JsonArg>> {
    let mut p = Json::new(json);
    p.ws();
    if p.bump() != Some(b'{') {
        return None;
    }
    p.ws();
    let mut args = Vec::new();
    while p.peek().is_some() && p.peek() != Some(b'}') {
        let Some(key) = json_string(&mut p) else {
            return None;
        };
        p.ws();
        if p.bump() != Some(b':') {
            return None;
        }
        p.ws();
        let (is_string, value) = if p.peek() == Some(b'"') {
            let Some(v) = json_string(&mut p) else {
                return None;
            };
            (true, v)
        } else {
            let Some(raw) = json_raw_value(&mut p) else {
                return None;
            };
            (false, json_minify_raw_value(&raw))
        };
        args.push(JsonArg {
            key,
            value,
            is_string,
            used: false,
        });
        p.ws();
        if p.peek() == Some(b',') {
            p.i += 1;
        }
        p.ws();
    }
    if p.peek() != Some(b'}') {
        return None;
    }
    Some(args)
}

/// The contents of `src` when it is exactly one JSON string literal, and None
/// otherwise.
///
/// The V4.1 oracle tolerates a stringified `arguments` value: it decodes the
/// field up to twice before it gives up (encoding.py:528-537), so a replay
/// whose arguments were encoded one level too deep still renders that object's
/// parameters.  On the wire `arguments` is already a string
/// (docs/ds41-dsml-conformance.md requirement 2.3.5), so that extra decode is
/// this unwrap.
pub fn json_string_contents(src: &str) -> Option<String> {
    let mut p = Json::new(src);
    p.ws();
    let text = json_string(&mut p)?;
    p.ws();
    if p.peek().is_some() {
        return None;
    }
    Some(text)
}
