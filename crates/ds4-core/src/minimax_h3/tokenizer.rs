//! The FL2VA prompt tokenizer: GPT-2 byte-level BPE over the released Qwen
//! vocabulary.
//!
//! Source of truth `/data/h3.c` at `eae5d54`:
//!
//! - `h3_tokenizer.c:526-575` `h3_tokenizer_load`: the spec check, then the
//!   inverse vocabulary and the byte codec.
//! - `h3_tokenizer.c:367-400` `h3_parse_model`: `type` is `BPE`, `unk_token`
//!   is present and null, `vocab` and `merges` are required.
//! - `h3_tokenizer.c:338-369` `h3_parse_merges`: a merge is `"left right"` or
//!   `["left", "right"]`; the array index is its rank.
//! - `h3_tokenizer.c:405-459` added tokens: `content` and `id` required, and
//!   `single_word`/`lstrip`/`rstrip`/`normalized` must not be enabled.
//! - `h3_tokenizer.c:459-477` the normalizer must be `NFC`.
//! - `h3_tokenizer.c:560-573` the byte codec: the printable set keeps its byte,
//!   every other byte maps to `256 + offset`.
//! - `h3_tokenizer.c:705-765` `h3_pretokenize`: NFC, then the Qwen2/GPT-2
//!   split (contractions, letter runs, single digits, punctuation, whitespace).
//! - `h3_tokenizer.c:766-818` `h3_bpe`: lowest-rank adjacent pair, merging
//!   every non-overlapping occurrence of it, until no pair has a rank.
//! - `h3_tokenizer.c:829-877` `h3_tokenizer_encode`: added tokens are matched
//!   in the raw text first (earliest, longest on a tie); an empty prompt pads
//!   with `PAD_TOKEN_ID`, or stays empty when `pad_empty` is false.
//! - `h3_tokenizer.c:892-938` `h3_tokenizer_decode`: added-token content is
//!   literal, vocabulary symbols decode through the byte table, and a result
//!   that is not valid UTF-8 becomes one `U+FFFD`.
//!
//! Named deviations, both measured by the tokenizer gate:
//!
//! - `serde_json` parses the 7 MiB `tokenizer.json` instead of the reference's
//!   hand-written parser; the accepted grammar is the same, and the rejected
//!   cases are unit tests.
//! - Unicode classes come from the `regex` crate (`\p{L}`, `\p{N}`,
//!   `\p{White_Space}`) instead of ICU `u_charType`/`u_isUWhiteSpace`
//!   (`h3_tokenizer.c:661-673`). Both derive from the same Unicode character
//!   database, but their release may differ; the corpus gate against the
//!   reference binary is what proves the split agrees on real prompts.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

/// `H3_PAD_TOKEN_ID` (`h3_tokenizer.h:7`).
pub const PAD_TOKEN_ID: u32 = 151_643;

/// `h3_pair_key` joins the two symbols with `\x1f` (`h3_tokenizer.c:308-316`).
const PAIR_SEPARATOR: char = '\x1f';

/// The merge key cannot collide with a byte-level symbol: every control byte
/// is mapped to `256 + offset` by the codec, so `\x1f` never appears in one.
const BYTE_SYMBOLS: usize = 324;

/// The bytes that keep their own code point (`h3_tokenizer.c:564-566`).
const VISIBLE: [(u8, u8); 3] = [(b'!', b'~'), (0xa1, 0xac), (0xae, 0xff)];

/// `h3_contraction` (`h3_tokenizer.c:689-705`), in list order.
const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

/// What an empty prompt becomes (`h3_tokenizer_encode`'s `pad_empty`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmptyPrompt {
    Pad,
    Nothing,
}

/// Why a tokenizer could not be loaded, or a prompt could not be encoded.
#[derive(Debug)]
pub enum TokenizerError {
    /// `cannot read tokenizer JSON` (`h3_tokenizer.c:533`).
    Io,
    /// `malformed tokenizer JSON`, from `serde_json`.
    Json,
    /// `unexpected tokenizer model specification` (`h3_tokenizer.c:399`).
    Model(&'static str),
    /// `unsupported added-token policy` (`h3_tokenizer.c:438`).
    Added(&'static str),
    /// `tokenizer normalizer is not NFC` (`h3_tokenizer.c:476`).
    Normalizer,
    /// A symbol the merge loop produced that the vocabulary does not hold.
    Symbol(String),
    /// `token ID is out of range` (`h3_tokenizer.c:904`).
    OutOfRange(u32),
    /// `unknown token ID` (`h3_tokenizer.c:912`).
    Unknown(u32),
    /// `invalid byte-level token` (`h3_tokenizer.c:920`).
    ByteToken(String),
}

impl TokenizerError {
    /// One line, for a gate's stderr.
    pub fn token(&self) -> String {
        match self {
            TokenizerError::Io => "cannot read tokenizer JSON".to_string(),
            TokenizerError::Json => "malformed tokenizer JSON".to_string(),
            TokenizerError::Model(detail) => {
                format!("unexpected tokenizer model specification: {detail}")
            }
            TokenizerError::Added(detail) => format!("unsupported added-token policy: {detail}"),
            TokenizerError::Normalizer => "tokenizer normalizer is not NFC".to_string(),
            TokenizerError::Symbol(symbol) => format!("symbol not in vocabulary: {symbol:?}"),
            TokenizerError::OutOfRange(id) => format!("token ID is out of range: {id}"),
            TokenizerError::Unknown(id) => format!("unknown token ID: {id}"),
            TokenizerError::ByteToken(symbol) => format!("invalid byte-level token: {symbol:?}"),
        }
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.token())
    }
}

impl std::error::Error for TokenizerError {}

/// The released vocabulary, its merge ranks, and the added tokens.
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    merges: HashMap<String, u32>,
    added: Vec<(String, u32)>,
    inverse: Vec<Option<String>>,
    added_inverse: Vec<Option<String>>,
    byte_encoder: [String; 256],
    byte_decoder: [Option<u8>; BYTE_SYMBOLS],
}

impl Tokenizer {
    /// Read one `tokenizer.json` and check it is the specification the
    /// reference accepts.
    pub fn load(path: &Path) -> Result<Tokenizer, TokenizerError> {
        let text = std::fs::read_to_string(path).map_err(|_| TokenizerError::Io)?;
        let root: Value = serde_json::from_str(&text).map_err(|_| TokenizerError::Json)?;

        let model = root.get("model").ok_or(TokenizerError::Model("model"))?;

        check_spec(&root, model)?;

        let vocab = read_vocab(model)?;
        let merges = read_merges(model)?;
        let added = read_added(&root)?;
        let (byte_encoder, byte_decoder) = byte_codec();

        let width = vocab
            .values()
            .copied()
            .chain(added.iter().map(|(_, id)| *id))
            .max()
            .map_or(0, |id| id as usize + 1);

        let mut inverse = vec![None; width];
        let mut added_inverse = vec![None; width];

        for (symbol, id) in &vocab {
            inverse[*id as usize] = Some(symbol.clone());
        }

        for (content, id) in &added {
            added_inverse[*id as usize] = Some(content.clone());
        }

        Ok(Tokenizer {
            vocab,
            merges,
            added,
            inverse,
            added_inverse,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Number of ids the inverse table covers, vocabulary and added tokens.
    pub fn width(&self) -> usize {
        self.inverse.len()
    }

    /// Encode one prompt. Added tokens are matched in the raw text before any
    /// normalization (`h3_tokenizer.c:857-871`).
    pub fn encode(&self, text: &str, empty: EmptyPrompt) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let Some((offset, length, id)) = self.added_match(text, start) else {
                break;
            };

            if offset > start {
                self.encode_plain(&text[start..offset], &mut ids)?;
            }

            ids.push(id);
            start = offset + length;
        }

        if start < text.len() {
            self.encode_plain(&text[start..], &mut ids)?;
        }

        if ids.is_empty() && empty == EmptyPrompt::Pad {
            ids.push(PAD_TOKEN_ID);
        }

        Ok(ids)
    }

    /// Decode ids back to bytes. A byte sequence that is not valid UTF-8
    /// becomes one `U+FFFD`, exactly as `h3_tokenizer.c:925-934` does.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();

        for id in ids {
            let index = *id as usize;

            if index >= self.inverse.len() {
                return Err(TokenizerError::OutOfRange(*id));
            }

            if let Some(content) = &self.added_inverse[index] {
                bytes.extend_from_slice(content.as_bytes());
                continue;
            }

            let symbol = self
                .inverse
                .get(index)
                .and_then(Option::as_ref)
                .ok_or(TokenizerError::Unknown(*id))?;

            for value in symbol.chars() {
                let byte = self
                    .byte_decoder
                    .get(value as usize)
                    .copied()
                    .flatten()
                    .ok_or_else(|| TokenizerError::ByteToken(symbol.clone()))?;
                bytes.push(byte);
            }
        }

        Ok(String::from_utf8(bytes).unwrap_or_else(|_| "\u{fffd}".to_string()))
    }

    /// Plain text between added tokens: pretokenize, then BPE each piece.
    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), TokenizerError> {
        for piece in pretokenize(text) {
            self.bpe(&piece, ids)?;
        }

        Ok(())
    }

    /// `h3_bpe` (`h3_tokenizer.c:766-818`): the lowest-rank adjacent pair
    /// wins, earliest on a tie, and every non-overlapping occurrence of that
    /// pair is merged before the next scan.
    fn bpe(&self, piece: &str, ids: &mut Vec<u32>) -> Result<(), TokenizerError> {
        let mut symbols: Vec<String> = piece
            .bytes()
            .map(|byte| self.byte_encoder[byte as usize].clone())
            .collect();

        while symbols.len() > 1 {
            let mut best_rank = u32::MAX;
            let mut best = None;

            for index in 0..symbols.len() - 1 {
                let rank = self
                    .merges
                    .get(&pair_key(&symbols[index], &symbols[index + 1]))
                    .copied();
                if let Some(rank) = rank {
                    if rank < best_rank {
                        best_rank = rank;
                        best = Some(index);
                    }
                }
            }

            let Some(best) = best else {
                break;
            };

            let left = symbols[best].clone();
            let right = symbols[best + 1].clone();

            let mut merged = Vec::with_capacity(symbols.len());
            let mut index = 0;

            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == left
                    && symbols[index + 1] == right
                {
                    merged.push(format!("{left}{right}"));
                    index += 2;
                } else {
                    merged.push(symbols[index].clone());
                    index += 1;
                }
            }

            symbols = merged;
        }

        for symbol in symbols {
            let id = self
                .vocab
                .get(&symbol)
                .ok_or_else(|| TokenizerError::Symbol(symbol.clone()))?;
            ids.push(*id);
        }

        Ok(())
    }

    /// `h3_added_match` (`h3_tokenizer.c:829-845`): earliest match in the
    /// remaining text, the longest one when two start at the same byte.
    fn added_match(&self, text: &str, start: usize) -> Option<(usize, usize, u32)> {
        let mut found: Option<(usize, usize, u32)> = None;

        for (content, id) in &self.added {
            let Some(relative) = text[start..].find(content.as_str()) else {
                continue;
            };

            let at = start + relative;
            let length = content.len();
            let better = match found {
                None => true,
                Some((offset, size, _)) => at < offset || (at == offset && length > size),
            };

            if better {
                found = Some((at, length, *id));
            }
        }

        found
    }
}

/// The three checks `h3_parse_root` refuses to proceed without
/// (`h3_tokenizer.c:380-401`, `:459-477`).
fn check_spec(root: &Value, model: &Value) -> Result<(), TokenizerError> {
    if model.get("type").and_then(Value::as_str) != Some("BPE") {
        return Err(TokenizerError::Model("type is not BPE"));
    }

    if !matches!(model.get("unk_token"), Some(Value::Null)) {
        return Err(TokenizerError::Model("unk_token is not null"));
    }

    let normalizer = root
        .get("normalizer")
        .ok_or(TokenizerError::Model("normalizer"))?;

    if normalizer.get("type").and_then(Value::as_str) != Some("NFC") {
        return Err(TokenizerError::Normalizer);
    }

    Ok(())
}

/// `h3_parse_vocab` (`h3_tokenizer.c:317-337`).
fn read_vocab(model: &Value) -> Result<HashMap<String, u32>, TokenizerError> {
    let object = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or(TokenizerError::Model("vocab"))?;

    let mut vocab = HashMap::with_capacity(object.len());

    for (symbol, value) in object {
        let id = value
            .as_u64()
            .filter(|id| *id <= u64::from(u32::MAX))
            .ok_or(TokenizerError::Model("vocab id"))?;
        vocab.insert(symbol.clone(), id as u32);
    }

    Ok(vocab)
}

/// `h3_parse_merges` (`h3_tokenizer.c:338-369`).
fn read_merges(model: &Value) -> Result<HashMap<String, u32>, TokenizerError> {
    let list = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or(TokenizerError::Model("merges"))?;

    let mut merges = HashMap::with_capacity(list.len());

    for (rank, entry) in list.iter().enumerate() {
        let (left, right) = match entry {
            Value::String(joined) => joined
                .split_once(' ')
                .ok_or(TokenizerError::Model("merge without a space"))?,
            Value::Array(pair) => {
                let [left, right] = pair.as_slice() else {
                    return Err(TokenizerError::Model("merge pair"));
                };
                left.as_str()
                    .zip(right.as_str())
                    .ok_or(TokenizerError::Model("merge pair"))?
            }
            _ => return Err(TokenizerError::Model("merge entry")),
        };

        merges.insert(pair_key(left, right), rank as u32);
    }

    Ok(merges)
}

/// `h3_parse_added` (`h3_tokenizer.c:405-459`): every policy flag the
/// reference refuses is refused here too, whether or not it is present.
fn read_added(root: &Value) -> Result<Vec<(String, u32)>, TokenizerError> {
    let Some(list) = root.get("added_tokens") else {
        return Ok(Vec::new());
    };
    let list = list
        .as_array()
        .ok_or(TokenizerError::Model("added_tokens"))?;

    let mut added = Vec::with_capacity(list.len());

    for entry in list {
        for policy in ["single_word", "lstrip", "rstrip", "normalized"] {
            if entry
                .get(policy)
                .is_some_and(|value| value.as_bool() != Some(false))
            {
                return Err(TokenizerError::Added(policy));
            }
        }

        let content = entry
            .get("content")
            .and_then(Value::as_str)
            .ok_or(TokenizerError::Model("added content"))?;
        let id = entry
            .get("id")
            .and_then(Value::as_u64)
            .filter(|id| *id <= u64::from(u32::MAX))
            .ok_or(TokenizerError::Model("added id"))?;

        added.push((content.to_string(), id as u32));
    }

    Ok(added)
}

/// The reference's merge key (`h3_tokenizer.c:308-316`).
fn pair_key(left: &str, right: &str) -> String {
    let mut key = String::with_capacity(left.len() + right.len() + 1);
    key.push_str(left);
    key.push(PAIR_SEPARATOR);
    key.push_str(right);
    key
}

/// `h3_tokenizer_load`'s byte codec (`h3_tokenizer.c:560-573`): printable
/// bytes keep their value, the other 68 bytes take `256 + offset`.
fn byte_codec() -> ([String; 256], [Option<u8>; BYTE_SYMBOLS]) {
    let mut encoder = std::array::from_fn(|_| String::new());
    let mut decoder = [None; BYTE_SYMBOLS];
    let mut extra = 0u32;

    for byte in 0..=255u8 {
        let code = if VISIBLE
            .iter()
            .any(|(first, last)| byte >= *first && byte <= *last)
        {
            u32::from(byte)
        } else {
            let code = 256 + extra;
            extra += 1;
            code
        };

        // Every code is a valid scalar value: 0..=255 or 256..=323.
        encoder[byte as usize] = char::from_u32(code)
            .expect("a valid code point")
            .to_string();
        decoder[code as usize] = Some(byte);
    }

    (encoder, decoder)
}

/// One decoded code point with its byte span.
#[derive(Clone, Copy)]
struct Point {
    value: char,
    offset: usize,
    length: usize,
}

/// `h3_pretokenize` (`h3_tokenizer.c:705-765`).
fn pretokenize(text: &str) -> Vec<String> {
    let normalized = normalize(text);
    let points = points(&normalized);
    let mut pieces = Vec::new();
    let mut index = 0;

    while index < points.len() {
        let stop = piece_end(&points, index);
        let first = points[index];
        let last = points[stop - 1];
        let begin = first.offset;
        let end = last.offset + last.length;

        pieces.push(normalized[begin..end].to_string());
        index = stop;
    }

    pieces
}

/// NFC, skipped for ASCII because ASCII is already normalized.
fn normalize(text: &str) -> String {
    if text.is_ascii() {
        text.to_string()
    } else {
        text.nfc().collect()
    }
}

fn points(text: &str) -> Vec<Point> {
    text.char_indices()
        .map(|(offset, value)| Point {
            value,
            offset,
            length: value.len_utf8(),
        })
        .collect()
}

/// The end of the piece that starts at `index` (`h3_tokenizer.c:713-757`).
fn piece_end(points: &[Point], index: usize) -> usize {
    let count = points.len();
    let contraction = contraction(points, index);

    if contraction > 0 {
        return index + contraction;
    }

    let value = points[index].value;
    let letter_start = if letter(value) {
        Some(index)
    } else if value != '\r'
        && value != '\n'
        && !number(value)
        && index + 1 < count
        && letter(points[index + 1].value)
    {
        Some(index + 1)
    } else {
        None
    };

    if let Some(start) = letter_start {
        let mut stop = start;
        while stop < count && letter(points[stop].value) {
            stop += 1;
        }
        return stop;
    }

    if number(value) {
        return index + 1;
    }

    punctuation_end(points, index)
}

/// The punctuation and whitespace arms (`h3_tokenizer.c:732-756`).
fn punctuation_end(points: &[Point], index: usize) -> usize {
    let count = points.len();
    let value = points[index].value;
    let next = points.get(index + 1).map(|point| point.value);
    let punct_start = index
        + usize::from(
            value == ' ' && next.is_some_and(|next| !space(next) && !letter(next) && !number(next)),
        );

    let mut stop = punct_start;
    while stop < count
        && !space(points[stop].value)
        && !letter(points[stop].value)
        && !number(points[stop].value)
    {
        stop += 1;
    }

    if stop > punct_start {
        while stop < count && (points[stop].value == '\r' || points[stop].value == '\n') {
            stop += 1;
        }
        return stop;
    }

    if !space(value) {
        return index + 1;
    }

    whitespace_end(points, index)
}

/// `\s*[\r\n]+` first, then a run that reaches the end, then all but the last
/// space, then one space (`h3_tokenizer.c:744-756`).
fn whitespace_end(points: &[Point], index: usize) -> usize {
    let count = points.len();
    let mut end = index + 1;

    while end < count && space(points[end].value) {
        end += 1;
    }

    let newline = (index..end)
        .rev()
        .find(|cursor| points[*cursor].value == '\r' || points[*cursor].value == '\n')
        .map(|cursor| cursor + 1);

    match newline {
        Some(stop) => stop,
        None if end == count => end,
        None if end - index > 1 => end - 1,
        None => index + 1,
    }
}

/// `h3_contraction` (`h3_tokenizer.c:689-705`): ASCII case-insensitive.
fn contraction(points: &[Point], index: usize) -> usize {
    if points[index].value != '\'' {
        return 0;
    }

    for value in CONTRACTIONS {
        let length = value.len();

        if index + length > points.len() {
            continue;
        }

        let matches = value.chars().enumerate().skip(1).all(|(offset, want)| {
            let mut got = points[index + offset].value;
            if got.is_ascii_uppercase() {
                got = got.to_ascii_lowercase();
            }
            got == want
        });

        if matches {
            return length;
        }
    }

    0
}

/// `h3_letter` (`h3_tokenizer.c:661-667`): general category L.
fn letter(value: char) -> bool {
    if value.is_ascii() {
        return value.is_ascii_alphabetic();
    }
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    class(&PATTERN, r"^\p{L}$", value)
}

/// `h3_number` (`h3_tokenizer.c:669-673`): Nd, Nl and No.
fn number(value: char) -> bool {
    if value.is_ascii() {
        return value.is_ascii_digit();
    }
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    class(&PATTERN, r"^\p{N}$", value)
}

/// `h3_space` (`h3_tokenizer.c:675-677`): Unicode `White_Space`, plus the
/// four separator controls ICU leaves out of it.
fn space(value: char) -> bool {
    if value.is_ascii() {
        return matches!(
            value,
            '\t' | '\n' | '\x0b' | '\x0c' | '\r' | ' ' | '\x1c'..='\x1f'
        );
    }
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    class(&PATTERN, r"^\p{White_Space}$", value)
}

/// One anchored class test. The pattern is compiled once per call site.
fn class(pattern: &OnceLock<Regex>, source: &str, value: char) -> bool {
    let regex = pattern.get_or_init(|| Regex::new(source).expect("a static pattern"));
    let mut buffer = [0u8; 4];

    regex.is_match(value.encode_utf8(&mut buffer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    /// A one-line specification with synthetic symbols and merges, so the
    /// model-free tests exercise the same loader the checkpoint uses.
    fn specification(extra_vocab: &[(&str, u32)], merges: &[&str], added: &str) -> String {
        let (encoder, _) = byte_codec();
        let mut vocab: Vec<String> = Vec::new();

        for byte in b"Heloi ,!Wrd:\x01" {
            vocab.push(format!("\"{}\":{}", encoder[*byte as usize], byte));
        }
        for (symbol, id) in extra_vocab {
            vocab.push(format!("\"{symbol}\":{id}"));
        }

        let merges: Vec<String> = merges.iter().map(|pair| format!("\"{pair}\"")).collect();

        format!(
            concat!(
                "{{\"normalizer\":{{\"type\":\"NFC\"}},",
                "\"model\":{{\"type\":\"BPE\",\"unk_token\":null,",
                "\"vocab\":{{{}}},\"merges\":[{}]}},",
                "\"added_tokens\":{}}}"
            ),
            vocab.join(","),
            merges.join(","),
            added
        )
    }

    fn load(specification: &str) -> Result<Tokenizer, TokenizerError> {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

        let ticket = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path: PathBuf =
            std::env::temp_dir().join(format!("h3-tokenizer-{}-{ticket}.json", std::process::id()));

        let mut file = std::fs::File::create(&path).expect("a temp file");
        file.write_all(specification.as_bytes()).expect("a write");
        let loaded = Tokenizer::load(&path);
        let _ = std::fs::remove_file(&path);

        loaded
    }

    const ADDED: &str = "[{\"content\":\"<|im_start|>\",\"id\":151644,\
                          \"single_word\":false,\"lstrip\":false,\
                          \"rstrip\":false,\"normalized\":false}]";

    fn fixture() -> Tokenizer {
        load(&specification(
            &[("He", 300), ("Hel", 301), ("Hell", 302), ("Hello", 303)],
            &["H e", "He l", "Hel l", "Hell o"],
            ADDED,
        ))
        .expect("a tokenizer")
    }

    #[test]
    fn invalid_byte_symbol_returns_error() {
        let tokenizer = load(&specification(&[("한", 300)], &[], "[]")).unwrap();
        assert!(matches!(
            tokenizer.decode(&[300]),
            Err(TokenizerError::ByteToken(symbol)) if symbol == "한"
        ));
    }

    #[test]
    fn rejects_invalid_merge_pair() {
        let model = serde_json::json!({"merges": [["H", "e", "ignored"]]});
        assert!(matches!(
            read_merges(&model),
            Err(TokenizerError::Model("merge pair"))
        ));
        assert!(read_merges(&serde_json::json!({"merges": [["H", "e"]]})).is_ok());
    }

    #[test]
    fn rejects_invalid_added_policy() {
        for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
            for value in [
                serde_json::Value::Null,
                serde_json::json!(1),
                serde_json::json!("false"),
            ] {
                let mut root = serde_json::json!({"added_tokens": [{"content": "x", "id": 9}]});
                root["added_tokens"][0][flag] = value;
                assert!(
                    matches!(read_added(&root), Err(TokenizerError::Added(name)) if name == flag)
                );
            }
        }
    }

    #[test]
    fn the_byte_codec_keeps_printable_bytes_and_offsets_the_rest() {
        let (encoder, decoder) = byte_codec();

        assert_eq!(encoder[b'A' as usize], "A");
        assert_eq!(encoder[b'~' as usize], "~");
        // Space is not printable, so it takes 256 + 32 (`h3_tokenizer.c:560-573`).
        assert_eq!(encoder[b' ' as usize], "\u{120}");
        assert_eq!(decoder[0x120], Some(b' '));
        // Byte 1 is not printable either, so it takes 256 + 1
        // (`h3_tokenizer.c:560-573`).
        assert_eq!(decoder[257], Some(1));
        assert_eq!(encoder.len(), 256);
    }

    #[test]
    fn the_splitter_matches_the_reference_arms() {
        let pieces = pretokenize("Hello, WORLD!  2026\n中文");

        assert_eq!(
            pieces,
            ["Hello", ",", " WORLD", "!", " ", " ", "2", "0", "2", "6", "\n", "中文"]
        );
    }

    #[test]
    fn a_contraction_is_its_own_piece_and_ignores_ascii_case() {
        assert_eq!(pretokenize("it's"), ["it", "'s"]);
        assert_eq!(pretokenize("IT'LL"), ["IT", "'LL"]);
    }

    #[test]
    fn whitespace_before_a_word_keeps_one_space() {
        assert_eq!(pretokenize("a  b"), ["a", " ", " b"]);
        assert_eq!(pretokenize("a   "), ["a", "   "]);
    }

    #[test]
    fn the_lowest_rank_pair_merges_first() {
        let tokenizer = fixture();

        assert_eq!(
            tokenizer.encode("Hello", EmptyPrompt::Nothing).unwrap(),
            [303]
        );
        assert_eq!(
            tokenizer.encode("Hello!", EmptyPrompt::Nothing).unwrap(),
            [303, b'!' as u32]
        );
    }

    #[test]
    fn added_tokens_win_over_the_merge_loop() {
        let tokenizer = fixture();

        assert_eq!(
            tokenizer
                .encode("Hi<|im_start|>", EmptyPrompt::Nothing)
                .unwrap(),
            [b'H' as u32, b'i' as u32, 151_644]
        );
    }

    #[test]
    fn an_empty_prompt_pads_only_when_asked() {
        let tokenizer = fixture();

        assert_eq!(
            tokenizer.encode("", EmptyPrompt::Pad).unwrap(),
            [PAD_TOKEN_ID]
        );
        assert_eq!(
            tokenizer.encode("", EmptyPrompt::Nothing).unwrap(),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn decode_reverses_the_byte_codec() {
        let tokenizer = fixture();

        assert_eq!(tokenizer.decode(&[303, b'!' as u32]).unwrap(), "Hello!");
        assert_eq!(tokenizer.decode(&[151_644]).unwrap(), "<|im_start|>");
    }

    #[test]
    fn decode_rejects_an_id_the_specification_does_not_hold() {
        let tokenizer = fixture();

        assert!(matches!(
            tokenizer.decode(&[151_644 + 1]),
            Err(TokenizerError::OutOfRange(_))
        ));
    }

    #[test]
    fn a_symbol_without_a_vocabulary_entry_fails_the_encode() {
        let tokenizer = load(&specification(&[], &["H e"], ADDED)).expect("a tokenizer");

        assert!(matches!(
            tokenizer.encode("Hello", EmptyPrompt::Nothing),
            Err(TokenizerError::Symbol(_))
        ));
    }

    #[test]
    fn the_portable_fixture_matches_the_reference_vectors() {
        // Byte-identical copy of `/data/h3.c` `tests/tokenizer_portable_fixture.json`
        // (sha256 4f300dcd...47e73); the cases are `tests/test_tokenizer_portable.c:34-47`.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/h3/tokenizer_portable.json");
        let tokenizer = Tokenizer::load(&path).expect("the portable fixture");

        for (text, expected) in [
            ("A A", vec![1, 3]),
            ("e\u{301}", vec![4]),
            ("<special>A", vec![10, 1]),
        ] {
            let ids = tokenizer.encode(text, EmptyPrompt::Nothing).unwrap();
            assert_eq!(ids, expected, "{text}");
        }

        assert_eq!(tokenizer.decode(&[1, 3]).unwrap(), "A A");
        // NFC composes the combining acute, then the byte codec reverses it.
        assert_eq!(tokenizer.decode(&[4]).unwrap(), "é");
        assert_eq!(tokenizer.decode(&[10, 1]).unwrap(), "<special>A");
        assert_eq!(
            tokenizer.encode("", EmptyPrompt::Pad).unwrap(),
            [PAD_TOKEN_ID]
        );
    }

    #[test]
    fn the_specification_refusals_match_the_reference() {
        let wordpiece =
            specification(&[], &[], ADDED).replace("\"type\":\"BPE\"", "\"type\":\"WordPiece\"");
        assert!(matches!(load(&wordpiece), Err(TokenizerError::Model(_))));

        let nfkc = specification(&[], &[], ADDED).replace("\"type\":\"NFC\"", "\"type\":\"NFKC\"");
        assert!(matches!(load(&nfkc), Err(TokenizerError::Normalizer)));

        let no_unk = specification(&[], &[], ADDED).replace("\"unk_token\":null,", "");
        assert!(matches!(load(&no_unk), Err(TokenizerError::Model(_))));

        let policy = specification(&[], &[], "[{\"content\":\"x\",\"id\":9,\"lstrip\":true}]");
        assert!(matches!(
            load(&policy),
            Err(TokenizerError::Added("lstrip"))
        ));
    }
}
