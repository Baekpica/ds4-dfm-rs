//! KVC file envelope: 48-byte header + text length + text + payload.
//!
//! Matches `ds4_kvstore_fill_header` / `ds4_kvstore_read_header` and the
//! store fwrite order in `ds4_kvstore.c`. Do not invent a new format.

use crate::sha1::sha1_hex;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const FIXED_HEADER: usize = 48;
pub const MAGIC: [u8; 3] = [b'K', b'V', b'C'];
pub const VERSION: u8 = 1;
pub const PAYLOAD_ABI: u8 = 2;

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

pub const EXT_TOOL_MAP: u8 = 1 << 0;
pub const EXT_RESPONSES_VISIBLE: u8 = 1 << 1;
pub const EXT_THINKING_VISIBLE: u8 = 1 << 2;
pub const EXT_SESSION_TITLE: u8 = 1 << 3;
pub const EXT_BANK_REPLAY_V1: u8 = 1 << 4;
pub const EXT_IMAGE_PIXELS_V2: u8 = 1 << 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Reason {
    Unknown = 0,
    Cold = 1,
    Continued = 2,
    Evict = 3,
    Shutdown = 4,
    AgentSystem = 5,
    AgentSession = 6,
    BankEvict = 7,
    BankShutdown = 8,
    BankCheckpoint = 9,
}

impl Reason {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Cold,
            2 => Self::Continued,
            3 => Self::Evict,
            4 => Self::Shutdown,
            5 => Self::AgentSystem,
            6 => Self::AgentSession,
            7 => Self::BankEvict,
            8 => Self::BankShutdown,
            9 => Self::BankCheckpoint,
            _ => Self::Unknown,
        }
    }

    pub fn from_name(name: &str) -> Self {
        match name {
            "cold" => Self::Cold,
            "continued" => Self::Continued,
            "evict" => Self::Evict,
            "shutdown" => Self::Shutdown,
            "agent-system" => Self::AgentSystem,
            "agent-session" => Self::AgentSession,
            "bank-evict" => Self::BankEvict,
            "bank-shutdown" => Self::BankShutdown,
            "bank-checkpoint" => Self::BankCheckpoint,
            _ => Self::Unknown,
        }
    }

    pub fn family(self) -> u8 {
        match self as u8 {
            1..=4 => 1,
            5..=6 => 2,
            7..=9 => 3,
            _ => 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub quant_bits: u8,
    pub reason: Reason,
    pub ext_flags: u8,
    pub model_id: u8,
    pub tokens: u32,
    pub hits: u32,
    pub ctx_size: u32,
    pub created_at: u64,
    pub last_used: u64,
    pub payload_bytes: u64,
    pub text_bytes: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub header: Header,
    pub text: Vec<u8>,
    pub payload: Vec<u8>,
    pub trailer: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Metadata {
    pub(crate) header: Header,
    pub(crate) payload_offset: u64,
    pub(crate) trailer_bytes: u64,
    pub(crate) file_size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Envelope {
    pub header: Header,
    pub text: Vec<u8>,
    pub payload_offset: u64,
    pub trailer_bytes: u64,
    pub file_size: u64,
}

#[derive(Debug)]
pub enum FormatError {
    Io(io::Error),
    BadMagic,
    BadVersion,
    BadPayloadAbi,
    BadQuant,
    ZeroTokens,
    Truncated,
    TextLenMismatch,
    TrailerTooLarge,
}

impl From<io::Error> for FormatError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::BadMagic => write!(f, "not a KVC file"),
            Self::BadVersion => write!(f, "unsupported KVC version"),
            Self::BadPayloadAbi => write!(f, "unsupported KVC payload ABI"),
            Self::BadQuant => write!(f, "quant_bits must be 2 or 4"),
            Self::ZeroTokens => write!(f, "token count is zero"),
            Self::Truncated => write!(f, "truncated KVC file"),
            Self::TextLenMismatch => write!(f, "text length does not match header"),
            Self::TrailerTooLarge => write!(f, "KVC trailer exceeds read bound"),
        }
    }
}

impl std::error::Error for FormatError {}

pub fn le_put32(p: &mut [u8], v: u32) {
    p[0] = v as u8;
    p[1] = (v >> 8) as u8;
    p[2] = (v >> 16) as u8;
    p[3] = (v >> 24) as u8;
}

pub fn le_get32(p: &[u8]) -> u32 {
    u32::from(p[0]) | (u32::from(p[1]) << 8) | (u32::from(p[2]) << 16) | (u32::from(p[3]) << 24)
}

pub fn le_put64(p: &mut [u8], v: u64) {
    for i in 0..8 {
        p[i] = (v >> (8 * i)) as u8;
    }
}

pub fn le_get64(p: &[u8]) -> u64 {
    let mut v = 0u64;
    for i in (0..8).rev() {
        v = (v << 8) | u64::from(p[i]);
    }
    v
}

pub fn fill_header(
    model_id: u8,
    quant_bits: u8,
    reason: Reason,
    ext_flags: u8,
    tokens: u32,
    hits: u32,
    ctx_size: u32,
    created_at: u64,
    last_used: u64,
    payload_bytes: u64,
) -> [u8; FIXED_HEADER] {
    let mut h = [0u8; FIXED_HEADER];
    h[0] = MAGIC[0];
    h[1] = MAGIC[1];
    h[2] = MAGIC[2];
    h[3] = VERSION;
    h[4] = quant_bits;
    h[5] = reason as u8;
    h[6] = ext_flags;
    h[7] = model_id;
    le_put32(&mut h[8..12], tokens);
    le_put32(&mut h[12..16], hits);
    le_put32(&mut h[16..20], ctx_size);
    h[20] = PAYLOAD_ABI;
    le_put64(&mut h[24..32], created_at);
    le_put64(&mut h[32..40], last_used);
    le_put64(&mut h[40..48], payload_bytes);
    h
}

pub fn parse_header(h: &[u8; FIXED_HEADER], text_bytes: u32) -> Result<Header, FormatError> {
    if h[0] != MAGIC[0] || h[1] != MAGIC[1] || h[2] != MAGIC[2] {
        return Err(FormatError::BadMagic);
    }
    if h[3] != VERSION {
        return Err(FormatError::BadVersion);
    }
    if h[20] != PAYLOAD_ABI {
        return Err(FormatError::BadPayloadAbi);
    }
    let tokens = le_get32(&h[8..12]);
    let quant_bits = h[4];
    if tokens == 0 {
        return Err(FormatError::ZeroTokens);
    }
    if quant_bits != 2 && quant_bits != 4 {
        return Err(FormatError::BadQuant);
    }
    Ok(Header {
        quant_bits,
        reason: Reason::from_u8(h[5]),
        ext_flags: h[6],
        model_id: h[7],
        tokens,
        hits: le_get32(&h[12..16]),
        ctx_size: le_get32(&h[16..20]),
        created_at: le_get64(&h[24..32]),
        last_used: le_get64(&h[32..40]),
        payload_bytes: le_get64(&h[40..48]),
        text_bytes,
    })
}

pub fn encode_file(record: &Record) -> Vec<u8> {
    let mut h = fill_header(
        record.header.model_id,
        record.header.quant_bits,
        record.header.reason,
        record.header.ext_flags,
        record.header.tokens,
        record.header.hits,
        record.header.ctx_size,
        record.header.created_at,
        record.header.last_used,
        record.payload.len() as u64,
    );
    // payload_bytes in the header follows the payload vector, matching C.
    le_put64(&mut h[40..48], record.payload.len() as u64);
    let mut out = Vec::with_capacity(
        FIXED_HEADER + 4 + record.text.len() + record.payload.len() + record.trailer.len(),
    );
    out.extend_from_slice(&h);
    let mut tb = [0u8; 4];
    le_put32(&mut tb, record.text.len() as u32);
    out.extend_from_slice(&tb);
    out.extend_from_slice(&record.text);
    out.extend_from_slice(&record.payload);
    out.extend_from_slice(&record.trailer);
    out
}

pub fn decode_file(bytes: &[u8]) -> Result<Record, FormatError> {
    if bytes.len() < FIXED_HEADER + 4 {
        return Err(FormatError::Truncated);
    }
    let mut hdr = [0u8; FIXED_HEADER];
    hdr.copy_from_slice(&bytes[..FIXED_HEADER]);
    let text_bytes = le_get32(&bytes[FIXED_HEADER..FIXED_HEADER + 4]);
    let header = parse_header(&hdr, text_bytes)?;
    let text_start = FIXED_HEADER + 4;
    let text_end = text_start
        .checked_add(text_bytes as usize)
        .ok_or(FormatError::Truncated)?;
    let payload_end = text_end
        .checked_add(header.payload_bytes as usize)
        .ok_or(FormatError::Truncated)?;
    if bytes.len() < payload_end {
        return Err(FormatError::Truncated);
    }
    Ok(Record {
        header,
        text: bytes[text_start..text_end].to_vec(),
        payload: bytes[text_end..payload_end].to_vec(),
        trailer: bytes[payload_end..].to_vec(),
    })
}

pub fn write_path(path: &Path, record: &Record) -> Result<(), FormatError> {
    let bytes = encode_file(record);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("kv.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.flush()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn write_stream(
    path: &Path,
    header: &Header,
    text: &[u8],
    payload: &mut impl Read,
    payload_bytes: u64,
    trailer: &[u8],
) -> Result<(), FormatError> {
    let tmp = stage_stream(path, header, text, payload, payload_bytes, trailer)?;
    let result = fs::rename(&tmp, path).map_err(FormatError::Io);
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

pub(crate) fn stage_stream(
    path: &Path,
    header: &Header,
    text: &[u8],
    payload: &mut impl Read,
    payload_bytes: u64,
    trailer: &[u8],
) -> Result<PathBuf, FormatError> {
    if text.len() > u32::MAX as usize {
        return Err(FormatError::TextLenMismatch);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let (tmp, mut f) = loop {
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_extension(format!("kv.tmp.{}.{}", std::process::id(), seq));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(f) => break (tmp, f),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(FormatError::Io(e)),
        }
    };
    let result: Result<(), FormatError> = (|| {
        let h = fill_header(
            header.model_id,
            header.quant_bits,
            header.reason,
            header.ext_flags,
            header.tokens,
            header.hits,
            header.ctx_size,
            header.created_at,
            header.last_used,
            payload_bytes,
        );
        f.write_all(&h)?;
        let mut raw_text_bytes = [0u8; 4];
        le_put32(&mut raw_text_bytes, text.len() as u32);
        f.write_all(&raw_text_bytes)?;
        f.write_all(text)?;
        let copied = io::copy(&mut payload.take(payload_bytes), &mut f)?;
        if copied != payload_bytes {
            return Err(FormatError::Truncated);
        }
        f.write_all(trailer)?;
        f.flush()?;
        drop(f);
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    Ok(tmp)
}

pub fn read_path(path: &Path) -> Result<Record, FormatError> {
    let mut f = fs::File::open(path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    decode_file(&bytes)
}

pub fn read_envelope(path: &Path) -> Result<Envelope, FormatError> {
    let (metadata, text) = read_text_prefix(path, usize::MAX)?;
    if text.len() != metadata.header.text_bytes as usize {
        return Err(FormatError::Truncated);
    }
    Ok(Envelope {
        header: metadata.header,
        text,
        payload_offset: metadata.payload_offset,
        trailer_bytes: metadata.trailer_bytes,
        file_size: metadata.file_size,
    })
}

pub fn read_trailer(path: &Path, max_bytes: u64) -> Result<(Header, Vec<u8>), FormatError> {
    let mut file = fs::File::open(path)?;
    let metadata = read_metadata_file(&mut file)?;
    if metadata.trailer_bytes > max_bytes {
        return Err(FormatError::TrailerTooLarge);
    }
    let trailer_start = metadata
        .payload_offset
        .checked_add(metadata.header.payload_bytes)
        .ok_or(FormatError::Truncated)?;
    let trailer_bytes =
        usize::try_from(metadata.trailer_bytes).map_err(|_| FormatError::TrailerTooLarge)?;
    file.seek(SeekFrom::Start(trailer_start))?;
    let mut trailer = vec![0; trailer_bytes];
    read_exact(&mut file, &mut trailer)?;
    Ok((metadata.header, trailer))
}

pub(crate) fn read_metadata(path: &Path) -> Result<Metadata, FormatError> {
    let mut f = fs::File::open(path)?;
    read_metadata_file(&mut f)
}

/// The header, and as much of the text as the file still holds, without
/// asking it for the payload behind them. A truncation can take the
/// payload and leave what says which conversation the record was keyed by.
pub(crate) fn read_header_text(
    path: &Path,
    max_bytes: usize,
) -> Result<(Header, Vec<u8>), FormatError> {
    let mut f = fs::File::open(path)?;
    let mut raw_header = [0u8; FIXED_HEADER];
    read_exact(&mut f, &mut raw_header)?;
    let mut raw_text_bytes = [0u8; 4];
    read_exact(&mut f, &mut raw_text_bytes)?;
    let header = parse_header(&raw_header, le_get32(&raw_text_bytes))?;
    let want = (header.text_bytes as usize).min(max_bytes);
    let mut text = Vec::new();
    f.take(want as u64).read_to_end(&mut text)?;
    Ok((header, text))
}

pub(crate) fn read_text_prefix(
    path: &Path,
    max_bytes: usize,
) -> Result<(Metadata, Vec<u8>), FormatError> {
    let mut f = fs::File::open(path)?;
    let metadata = read_metadata_file(&mut f)?;
    let want = (metadata.header.text_bytes as usize).min(max_bytes);
    let mut text = Vec::with_capacity(want);
    f.take(want as u64).read_to_end(&mut text)?;
    Ok((metadata, text))
}

fn read_metadata_file(f: &mut fs::File) -> Result<Metadata, FormatError> {
    let file_size = f.metadata()?.len();
    if file_size < (FIXED_HEADER + 4) as u64 {
        return Err(FormatError::Truncated);
    }

    let mut raw_header = [0u8; FIXED_HEADER];
    read_exact(f, &mut raw_header)?;
    let mut raw_text_bytes = [0u8; 4];
    read_exact(f, &mut raw_text_bytes)?;
    let text_bytes = le_get32(&raw_text_bytes);
    let header = parse_header(&raw_header, text_bytes)?;
    let payload_offset = (FIXED_HEADER + 4) as u64 + u64::from(text_bytes);
    let payload_end = payload_offset
        .checked_add(header.payload_bytes)
        .ok_or(FormatError::Truncated)?;
    if file_size < payload_end {
        return Err(FormatError::Truncated);
    }

    Ok(Metadata {
        header,
        payload_offset,
        trailer_bytes: file_size - payload_end,
        file_size,
    })
}

fn read_exact(f: &mut fs::File, buf: &mut [u8]) -> Result<(), FormatError> {
    match f.read_exact(buf) {
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Err(FormatError::Truncated),
        result => result.map_err(FormatError::Io),
    }
}

pub fn text_sha_hex(text: &[u8]) -> String {
    sha1_hex(text)
}

pub fn sha_hex_name(name: &str) -> Option<String> {
    if name.len() != 43 || !name.ends_with(".kv") {
        return None;
    }
    let hex = &name[..40];
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some(hex.to_ascii_lowercase())
}

pub fn path_for_sha(dir: &Path, sha: &str) -> PathBuf {
    dir.join(format!("{sha}.kv"))
}

pub fn key_kind(ext_flags: u8) -> &'static str {
    if ext_flags & EXT_RESPONSES_VISIBLE != 0 {
        "responses-visible"
    } else if ext_flags & EXT_THINKING_VISIBLE != 0 {
        "thinking-visible"
    } else {
        "token-text"
    }
}

pub fn is_bank_replay_v1(reason: Reason, ext_flags: u8) -> bool {
    reason.family() == 3 && ext_flags & EXT_BANK_REPLAY_V1 != 0
}

pub fn is_automatic_exact_replay(reason: Reason, ext_flags: u8) -> bool {
    match reason.family() {
        1 => ext_flags & EXT_BANK_REPLAY_V1 == 0,
        _ => is_bank_replay_v1(reason, ext_flags),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ErrorAfterPrefix {
        offset: usize,
    }

    impl Read for ErrorAfterPrefix {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            const PREFIX: &[u8] = b"DSV4";
            if self.offset == PREFIX.len() {
                return Err(io::Error::other("injected payload read failure"));
            }
            let n = buf.len().min(PREFIX.len() - self.offset);
            buf[..n].copy_from_slice(&PREFIX[self.offset..self.offset + n]);
            self.offset += n;
            Ok(n)
        }
    }

    fn sample() -> Record {
        Record {
            header: Header {
                quant_bits: 2,
                reason: Reason::Cold,
                ext_flags: 0,
                model_id: 0,
                tokens: 512,
                hits: 3,
                ctx_size: 2048,
                created_at: 1_700_000_000,
                last_used: 1_700_000_100,
                payload_bytes: 4,
                text_bytes: 5,
            },
            text: b"hello".to_vec(),
            payload: b"ABCD".to_vec(),
            trailer: Vec::new(),
        }
    }

    #[test]
    fn header_round_trip() {
        let rec = sample();
        let bytes = encode_file(&rec);
        let got = decode_file(&bytes).unwrap();
        assert_eq!(got.text, rec.text);
        assert_eq!(got.payload, rec.payload);
        assert_eq!(got.header.tokens, 512);
        assert_eq!(got.header.quant_bits, 2);
        assert_eq!(got.header.reason, Reason::Cold);
        assert_eq!(got.header.ctx_size, 2048);
        assert_eq!(&bytes[..3], b"KVC");
        assert_eq!(bytes[3], 1);
        assert_eq!(bytes[20], 2);
    }

    #[test]
    fn envelope_has_offsets_without_payload() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-envelope-{}", std::process::id()));
        let path = dir.join("sample.kv");
        let _ = fs::remove_dir_all(&dir);
        let mut rec = sample();
        rec.trailer = b"end".to_vec();
        write_path(&path, &rec).unwrap();

        let got = read_envelope(&path).unwrap();
        assert_eq!(got.header, rec.header);
        assert_eq!(got.text, rec.text);
        assert_eq!(
            got.payload_offset,
            (FIXED_HEADER + 4 + rec.text.len()) as u64
        );
        assert_eq!(got.trailer_bytes, rec.trailer.len() as u64);
        assert_eq!(got.file_size, fs::metadata(&path).unwrap().len());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn trailer_read_seeks_past_sparse_payload_and_honors_bound() {
        use std::io::{Seek, SeekFrom};

        const PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;
        let dir =
            std::env::temp_dir().join(format!("ds4-kv-trailer-sparse-{}", std::process::id()));
        let path = dir.join("sample.kv");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let rec = sample();
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(&fill_header(
            rec.header.model_id,
            rec.header.quant_bits,
            rec.header.reason,
            EXT_TOOL_MAP,
            rec.header.tokens,
            rec.header.hits,
            rec.header.ctx_size,
            rec.header.created_at,
            rec.header.last_used,
            PAYLOAD_BYTES,
        ))
        .unwrap();
        let mut text_bytes = [0u8; 4];
        le_put32(&mut text_bytes, rec.text.len() as u32);
        file.write_all(&text_bytes).unwrap();
        file.write_all(&rec.text).unwrap();
        file.seek(SeekFrom::Current(PAYLOAD_BYTES as i64)).unwrap();
        file.write_all(b"\0\xffTR").unwrap();
        drop(file);

        let (header, trailer) = read_trailer(&path, 4).unwrap();
        assert_eq!(header.payload_bytes, PAYLOAD_BYTES);
        assert_eq!(header.ext_flags, EXT_TOOL_MAP);
        assert_eq!(trailer, b"\0\xffTR");
        assert!(matches!(
            read_trailer(&path, 3),
            Err(FormatError::TrailerTooLarge)
        ));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_wrong_quant() {
        let mut rec = sample();
        rec.header.quant_bits = 8;
        let bytes = encode_file(&rec);
        assert!(matches!(decode_file(&bytes), Err(FormatError::BadQuant)));
    }

    #[test]
    fn filename_is_text_sha() {
        assert_eq!(sha_hex_name("not-a-kv"), None);
        let sha = text_sha_hex(b"hello");
        assert_eq!(sha.len(), 40);
        assert_eq!(
            sha_hex_name(&format!("{sha}.kv")).as_deref(),
            Some(sha.as_str())
        );
    }

    #[test]
    fn stream_failure_preserves_destination_and_removes_temp() {
        let dir =
            std::env::temp_dir().join(format!("ds4-kv-stream-failure-{}", std::process::id()));
        let path = dir.join("entry.kv");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, b"original KVC bytes").unwrap();

        let rec = sample();
        let mut payload = ErrorAfterPrefix { offset: 0 };
        let result = write_stream(&path, &rec.header, &rec.text, &mut payload, 8, &rec.trailer);

        assert!(result.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original KVC bytes");
        let mut names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names, [std::ffi::OsString::from("entry.kv")]);

        let mut short_payload = &b"DSV4"[..];
        let result = write_stream(
            &path,
            &rec.header,
            &rec.text,
            &mut short_payload,
            8,
            &rec.trailer,
        );
        assert!(result.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original KVC bytes");
        let names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, [std::ffi::OsString::from("entry.kv")]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stream_layout_matches_buffered_encoder() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-stream-layout-{}", std::process::id()));
        let path = dir.join("entry.kv");
        let _ = fs::remove_dir_all(&dir);

        let mut rec = sample();
        rec.trailer = b"KVT1 trailer".to_vec();
        let mut payload = rec.payload.as_slice();
        write_stream(
            &path,
            &rec.header,
            &rec.text,
            &mut payload,
            rec.payload.len() as u64,
            &rec.trailer,
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), encode_file(&rec));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn staged_stream_leaves_destination_until_commit() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-stream-stage-{}", std::process::id()));
        let path = dir.join("entry.kv");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, b"original KVC bytes").unwrap();

        let rec = sample();
        let mut payload = rec.payload.as_slice();
        let staged = stage_stream(
            &path,
            &rec.header,
            &rec.text,
            &mut payload,
            rec.payload.len() as u64,
            &rec.trailer,
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"original KVC bytes");
        assert_eq!(fs::read(&staged).unwrap(), encode_file(&rec));

        fs::rename(staged, &path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), encode_file(&rec));
        let _ = fs::remove_dir_all(&dir);
    }
}
