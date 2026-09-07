//! Host-owned DSV4 session payload prefix: 13×u32 LE header + token ids.
//! GPU / logits / family tensor tails stay native (`ds4_session_save_payload`).
//!
//! Copied from `ds4.c` / `ds4.h` at v0.6.3-dfm. Little-endian `payload_put_u32`.

use std::io::{self, Read, Seek, SeekFrom};

use crate::session::SessionLedger;
use crate::shape::ModelFamily;

pub const MAGIC: u32 = 0x3456_5344; /* "DSV4" */
pub const VERSION: u32 = 3;
pub const U32_FIELDS: usize = 13;
pub const HEADER_BYTES: usize = U32_FIELDS * 4;

pub const LAYOUT_SOLAR: u32 = 0x3352_4C53; /* "SLR3" */
pub const LAYOUT_EXAONE: u32 = 0x3341_5845; /* "EXA3" */
pub const LAYOUT_MOTIF3: u32 = 0x3346_544D; /* "MTF3" */
pub const LAYOUT_DOTS3: u32 = 0x3353_5444; /* "DTS3" */
pub const LAYOUT_QWEN4EXP: u32 = 0x334e_5751; /* "QWN3" */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadLayout {
    DeepSeek,
    Solar,
    Exaone,
    Motif3,
    Dots3,
    Qwen4Exp,
}

impl PayloadLayout {
    pub fn from_fields(h: &[u32; U32_FIELDS]) -> Self {
        match h[5] {
            LAYOUT_SOLAR => Self::Solar,
            LAYOUT_EXAONE => Self::Exaone,
            LAYOUT_MOTIF3 => Self::Motif3,
            LAYOUT_DOTS3 => Self::Dots3,
            LAYOUT_QWEN4EXP => Self::Qwen4Exp,
            _ => Self::DeepSeek,
        }
    }

    pub fn family(self) -> ModelFamily {
        match self {
            Self::DeepSeek => ModelFamily::DeepSeek4,
            Self::Solar => ModelFamily::SolarOpen2,
            Self::Exaone => ModelFamily::ExaoneMoe,
            Self::Motif3 => ModelFamily::Motif3,
            Self::Dots3 => ModelFamily::Dots3Note,
            Self::Qwen4Exp => ModelFamily::Qwen4Exp,
        }
    }

    fn exceeds_context(self, tokens: usize, ctx: i32) -> bool {
        // Native QWN3 can persist an exactly full context.
        ctx <= 0 || tokens > ctx as usize || (tokens == ctx as usize && self != Self::Qwen4Exp)
    }

    pub fn oracle_name(self) -> &'static str {
        self.family().oracle_name()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostPrefix {
    pub fields: [u32; U32_FIELDS],
    pub tokens: Vec<u32>,
}

impl HostPrefix {
    pub fn layout(&self) -> PayloadLayout {
        PayloadLayout::from_fields(&self.fields)
    }

    pub fn token_count(&self) -> u32 {
        self.fields[7]
    }

    pub fn ctx(&self) -> u32 {
        self.fields[2]
    }

    pub fn version(&self) -> u32 {
        self.fields[1]
    }

    pub fn prefix_len(&self) -> usize {
        HEADER_BYTES + self.tokens.len() * 4
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.prefix_len());
        for f in self.fields {
            out.extend_from_slice(&put_u32(f));
        }
        for t in &self.tokens {
            out.extend_from_slice(&put_u32(*t));
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PayloadError {
    pub message: &'static str,
}

impl std::fmt::Display for PayloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for PayloadError {}

fn err(message: &'static str) -> PayloadError {
    PayloadError { message }
}

pub fn put_u32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

pub fn get_u32(in4: &[u8]) -> u32 {
    u32::from_le_bytes([in4[0], in4[1], in4[2], in4[3]])
}

pub fn encode_fields(fields: &[u32; U32_FIELDS], tokens: &[u32]) -> Vec<u8> {
    HostPrefix {
        fields: *fields,
        tokens: tokens.to_vec(),
    }
    .encode()
}

fn parse_fields(bytes: &[u8]) -> Result<[u32; U32_FIELDS], PayloadError> {
    if bytes.len() < HEADER_BYTES {
        return Err(err("truncated session payload"));
    }
    let mut fields = [0u32; U32_FIELDS];
    for i in 0..U32_FIELDS {
        let off = i * 4;
        fields[i] = get_u32(&bytes[off..off + 4]);
    }
    if fields[0] != MAGIC || fields[1] == 0 || fields[1] > VERSION {
        return Err(err("unsupported session payload version"));
    }
    Ok(fields)
}

pub fn parse_prefix(bytes: &[u8]) -> Result<HostPrefix, PayloadError> {
    let fields = parse_fields(bytes)?;
    let n = fields[7] as usize;
    let need = HEADER_BYTES.saturating_add(n.saturating_mul(4));
    if bytes.len() < need {
        return Err(err("truncated session payload"));
    }
    let mut tokens = Vec::with_capacity(n);
    for i in 0..n {
        let off = HEADER_BYTES + i * 4;
        tokens.push(get_u32(&bytes[off..off + 4]));
    }
    let prefix = HostPrefix { fields, tokens };
    validate_layout(&prefix)?;
    Ok(prefix)
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn read_payload_exact(reader: &mut impl Read, bytes: &mut [u8]) -> io::Result<()> {
    reader.read_exact(bytes).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            invalid_data("truncated session payload")
        } else {
            e
        }
    })
}

pub(crate) fn read_prefix_range(
    reader: &mut (impl Read + Seek),
    offset: u64,
    length: u64,
    family: ModelFamily,
    ctx: i32,
) -> io::Result<HostPrefix> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| invalid_data("session payload range overflows"))?;
    if end > reader.seek(SeekFrom::End(0))? {
        return Err(invalid_data("truncated session payload range"));
    }
    if length < HEADER_BYTES as u64 {
        return Err(invalid_data("truncated session payload"));
    }
    reader.seek(SeekFrom::Start(offset))?;
    let mut header = [0u8; HEADER_BYTES];
    read_payload_exact(reader, &mut header)?;
    let fields = parse_fields(&header).map_err(|e| invalid_data(e.message))?;
    if PayloadLayout::from_fields(&fields).family() != family {
        return Err(invalid_data(
            "session payload was written for a different model family",
        ));
    }
    let n = fields[7] as usize;
    if PayloadLayout::from_fields(&fields).exceeds_context(n, ctx) {
        return Err(invalid_data("session payload exceeds context"));
    }
    let prefix_len = HEADER_BYTES as u64 + fields[7] as u64 * 4;
    if prefix_len > length {
        return Err(invalid_data("truncated session payload"));
    }
    let token_bytes = n
        .checked_mul(4)
        .ok_or_else(|| invalid_data("session payload range overflows"))?;
    let mut raw_tokens = vec![0u8; token_bytes];
    read_payload_exact(reader, &mut raw_tokens)?;
    let tokens = raw_tokens.chunks_exact(4).map(get_u32).collect();
    let prefix = HostPrefix { fields, tokens };
    validate_layout(&prefix).map_err(|e| invalid_data(e.message))?;
    Ok(prefix)
}

fn validate_layout(p: &HostPrefix) -> Result<(), PayloadError> {
    match p.layout() {
        PayloadLayout::Solar
        | PayloadLayout::Exaone
        | PayloadLayout::Motif3
        | PayloadLayout::Dots3 => {
            if p.fields[12] != p.fields[7] {
                return Err(err("session payload token count does not match live rows"));
            }
        }
        // QWN3 word 12 is the PLE convolution byte count, not live rows.
        PayloadLayout::DeepSeek | PayloadLayout::Qwen4Exp => {}
    }
    Ok(())
}

pub fn tail(bytes: &[u8]) -> Result<&[u8], PayloadError> {
    let p = parse_prefix(bytes)?;
    Ok(&bytes[p.prefix_len()..])
}

impl SessionLedger {
    pub fn apply_payload(&mut self, prefix: &HostPrefix) -> Result<(), PayloadError> {
        if prefix.layout().family() != self.family {
            return Err(err(
                "session payload was written for a different model family",
            ));
        }
        if prefix
            .layout()
            .exceeds_context(prefix.tokens.len(), self.ctx)
        {
            return Err(err("session payload exceeds context"));
        }
        let tokens: Vec<i32> = prefix.tokens.iter().map(|&t| t as i32).collect();
        self.replace_checkpoint(&tokens);
        Ok(())
    }
}

fn fixture_deepseek() -> HostPrefix {
    HostPrefix {
        fields: [
            MAGIC, VERSION, 8192, 64, 8192, 512, 2048, 3, 61, 128, 64, 129280, 3,
        ],
        tokens: vec![10, 20, 30],
    }
}

fn fixture_cpu() -> HostPrefix {
    HostPrefix {
        fields: [
            MAGIC, VERSION, 8192, 64, 8192, 8192, 2048, 2, 61, 128, 64, 129280, 2,
        ],
        tokens: vec![1, 2],
    }
}

fn fixture_solar() -> HostPrefix {
    HostPrefix {
        fields: [
            MAGIC,
            VERSION,
            4096,
            1,
            4096,
            LAYOUT_SOLAR,
            1000,
            2,
            48,
            256,
            12,
            200064,
            2,
        ],
        tokens: vec![7, 8],
    }
}

fn fixture_exaone() -> HostPrefix {
    HostPrefix {
        fields: [
            MAGIC,
            VERSION,
            8192,
            64,
            48,
            LAYOUT_EXAONE,
            512,
            2,
            49,
            128,
            128,
            153088,
            2,
        ],
        tokens: vec![1, 2],
    }
}

fn fixture_motif3() -> HostPrefix {
    HostPrefix {
        fields: [
            MAGIC,
            VERSION,
            8192,
            2048,
            512,
            LAYOUT_MOTIF3,
            1024,
            2,
            53,
            64,
            128,
            151936,
            2,
        ],
        tokens: vec![3, 4],
    }
}

fn fixture_dots3() -> HostPrefix {
    HostPrefix {
        fields: [
            MAGIC,
            VERSION,
            8192,
            2048,
            512,
            LAYOUT_DOTS3,
            1024,
            2,
            46,
            64,
            128,
            152064,
            2,
        ],
        tokens: vec![5, 6],
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2 + 1);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s.push('\n');
    s
}

fn inspect_text(p: &HostPrefix) -> String {
    format!(
        "layout={} version={} ctx={} ntok={} prefix={}\ntokens={}\n",
        p.layout().oracle_name(),
        p.version(),
        p.ctx(),
        p.token_count(),
        p.prefix_len(),
        p.tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn parse_hex(s: &str) -> Result<Vec<u8>, PayloadError> {
    let raw: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    if raw.len() % 2 != 0 {
        return Err(err("truncated session payload"));
    }
    let mut out = Vec::with_capacity(raw.len() / 2);
    let bytes = raw.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, PayloadError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(err("truncated session payload")),
    }
}

fn flip_magic(mut bytes: Vec<u8>) -> Vec<u8> {
    if !bytes.is_empty() {
        bytes[0] ^= 0xff;
    }
    bytes
}

fn set_version(mut bytes: Vec<u8>, v: u32) -> Vec<u8> {
    if bytes.len() >= 8 {
        bytes[4..8].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Tape matching `tests/parity/payload_c_oracle`.
pub fn dump_script(name: &str) -> String {
    match name {
        "encode-deepseek" => hex(&fixture_deepseek().encode()),
        "encode-cpu" => hex(&fixture_cpu().encode()),
        "encode-solar" => hex(&fixture_solar().encode()),
        "encode-exaone" => hex(&fixture_exaone().encode()),
        "encode-motif3" => hex(&fixture_motif3().encode()),
        "encode-dots3" => hex(&fixture_dots3().encode()),
        "inspect-deepseek" => inspect_text(&fixture_deepseek()),
        "inspect-solar" => inspect_text(&fixture_solar()),
        "inspect-exaone" => inspect_text(&fixture_exaone()),
        "inspect-motif3" => inspect_text(&fixture_motif3()),
        "inspect-dots3" => inspect_text(&fixture_dots3()),
        "tail-offset" => format!("prefix={}\n", fixture_deepseek().prefix_len()),
        "reject-trunc" => match parse_prefix(&[0u8; 10]) {
            Ok(_) => "ok\n".into(),
            Err(e) => format!("ERROR {}\n", e.message),
        },
        "reject-magic" => match parse_prefix(&flip_magic(fixture_deepseek().encode())) {
            Ok(_) => "ok\n".into(),
            Err(e) => format!("ERROR {}\n", e.message),
        },
        "reject-version-0" => match parse_prefix(&set_version(fixture_deepseek().encode(), 0)) {
            Ok(_) => "ok\n".into(),
            Err(e) => format!("ERROR {}\n", e.message),
        },
        "reject-version-4" => match parse_prefix(&set_version(fixture_deepseek().encode(), 4)) {
            Ok(_) => "ok\n".into(),
            Err(e) => format!("ERROR {}\n", e.message),
        },
        "apply-deepseek" => {
            let p = fixture_deepseek();
            let mut host = SessionLedger::new(
                ModelFamily::DeepSeek4,
                crate::session::SessionBackend::Cuda,
                8192,
                64,
            );
            match host.apply_payload(&p) {
                Ok(()) => format!(
                    "APPLY valid={} pos={} tokens={}\n",
                    u32::from(host.valid),
                    host.pos(),
                    host.tokens()
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                Err(e) => format!("ERROR {}\n", e.message),
            }
        }
        "apply-family-miss" => {
            let p = fixture_solar();
            let mut host = SessionLedger::new(
                ModelFamily::DeepSeek4,
                crate::session::SessionBackend::Cuda,
                8192,
                64,
            );
            match host.apply_payload(&p) {
                Ok(()) => "ok\n".into(),
                Err(e) => format!("ERROR {}\n", e.message),
            }
        }
        _ => "ERROR unknown-script\n".into(),
    }
}

pub fn dump_cmd(cmd: &str, args: &[&str]) -> String {
    if cmd == "inspect" {
        if let Some(hexs) = args.first() {
            return match parse_hex(hexs).and_then(|b| parse_prefix(&b)) {
                Ok(p) => inspect_text(&p),
                Err(e) => format!("ERROR {}\n", e.message),
            };
        }
    }
    dump_script(cmd)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Seek};

    use super::*;

    #[test]
    fn qwen_full_context_payload() {
        let mut prefix = fixture_deepseek();
        prefix.fields[2] = 3;
        prefix.fields[5] = LAYOUT_QWEN4EXP;
        prefix.fields[12] = 368_640;
        let bytes = prefix.encode();
        // Both the file reader and host ledger must accept native's full
        // Qwen frontier, while still rejecting an overfull/invalid context.
        for (ctx, expected) in [(3, true), (2, false), (0, false), (-1, false)] {
            let read = read_prefix_range(
                &mut Cursor::new(&bytes),
                0,
                bytes.len() as u64,
                ModelFamily::Qwen4Exp,
                ctx,
            );
            let mut host = SessionLedger::new(
                ModelFamily::Qwen4Exp,
                crate::session::SessionBackend::Cuda,
                ctx,
                3,
            );
            assert_eq!(
                (read.is_ok(), host.apply_payload(&prefix).is_ok()),
                (expected, expected),
                "ctx={ctx}"
            );
        }
        let prefix = fixture_deepseek();
        let bytes = prefix.encode();
        assert!(read_prefix_range(
            &mut Cursor::new(&bytes),
            0,
            bytes.len() as u64,
            ModelFamily::DeepSeek4,
            3,
        )
        .is_err());
        let mut host = SessionLedger::new(
            ModelFamily::DeepSeek4,
            crate::session::SessionBackend::Cuda,
            3,
            3,
        );
        assert!(host.apply_payload(&prefix).is_err());
    }

    #[test]
    fn qwen_prefix_restores_host_ledger() {
        // Native QWN3 stores PLE convolution bytes in word 12, not live rows.
        let prefix = HostPrefix {
            fields: [
                MAGIC,
                VERSION,
                2048,
                2048,
                12,
                u32::from_le_bytes(*b"QWN3"),
                2048,
                3,
                48,
                4,
                128,
                248320,
                368_640,
            ],
            tokens: vec![10, 20, 30],
        };
        let mut bytes = vec![0xaa, 0xbb, 0xcc];
        bytes.extend_from_slice(&prefix.encode());
        bytes.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        let length = bytes.len() as u64 - 3;
        let mut cursor = Cursor::new(bytes.clone());
        let parsed =
            read_prefix_range(&mut cursor, 3, length, ModelFamily::Qwen4Exp, 2048).unwrap();
        assert_eq!(parsed, prefix);
        assert_eq!(
            cursor.stream_position().unwrap(),
            3 + prefix.prefix_len() as u64
        );
        assert_eq!(parse_prefix(&prefix.encode()).unwrap(), prefix);
        let mut host = SessionLedger::new(
            ModelFamily::Qwen4Exp,
            crate::session::SessionBackend::Cuda,
            2048,
            2048,
        );
        host.apply_payload(&parsed).unwrap();
        assert_eq!(host.tokens(), &[10, 20, 30]);

        // A valid Qwen prefix must never enter another family's ledger.
        let mut wrong = Cursor::new(bytes.clone());
        assert_eq!(
            read_prefix_range(&mut wrong, 3, length, ModelFamily::DeepSeek4, 2048,)
                .unwrap_err()
                .to_string(),
            "session payload was written for a different model family"
        );
        let mut short = Cursor::new(bytes);
        assert_eq!(
            read_prefix_range(
                &mut short,
                3,
                prefix.prefix_len() as u64 - 1,
                ModelFamily::Qwen4Exp,
                2048,
            )
            .unwrap_err()
            .to_string(),
            "truncated session payload"
        );
    }

    #[test]
    fn range_reader_validates_prefix_without_reading_native_tail() {
        let prefix = fixture_deepseek();
        let mut bytes = vec![0xaa, 0xbb, 0xcc];
        bytes.extend_from_slice(&prefix.encode());
        bytes.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        let range_len = (bytes.len() - 3) as u64;

        let mut cursor = Cursor::new(bytes.clone());
        let parsed =
            read_prefix_range(&mut cursor, 3, range_len, ModelFamily::DeepSeek4, 8192).unwrap();
        assert_eq!(parsed, prefix);
        assert_eq!(
            cursor.stream_position().unwrap(),
            3 + prefix.prefix_len() as u64
        );

        let mut cursor = Cursor::new(bytes.clone());
        let err = read_prefix_range(
            &mut cursor,
            3,
            prefix.prefix_len() as u64 - 1,
            ModelFamily::DeepSeek4,
            8192,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "truncated session payload");

        let mut cursor = Cursor::new(bytes.clone());
        let err = read_prefix_range(&mut cursor, 3, range_len + 1, ModelFamily::DeepSeek4, 8192)
            .unwrap_err();
        assert_eq!(err.to_string(), "truncated session payload range");

        let mut cursor = Cursor::new(bytes.clone());
        let err =
            read_prefix_range(&mut cursor, u64::MAX, 2, ModelFamily::DeepSeek4, 8192).unwrap_err();
        assert_eq!(err.to_string(), "session payload range overflows");

        let mut cursor = Cursor::new(bytes.clone());
        let err = read_prefix_range(
            &mut cursor,
            3,
            range_len,
            ModelFamily::DeepSeek4,
            prefix.token_count() as i32,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "session payload exceeds context");

        let mut cursor = Cursor::new(bytes);
        let err = read_prefix_range(&mut cursor, 3, range_len, ModelFamily::SolarOpen2, 8192)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "session payload was written for a different model family"
        );
        assert_eq!(cursor.stream_position().unwrap(), 3 + HEADER_BYTES as u64);
    }
}
