//! Spark-X2.6-4B-Base input adapter. The source `tokenizer.json` in the
//! checkpoint declares exactly:
//!
//!   normalizer     Sequence with no normalizers
//!   pre_tokenizer  Split(\p{N}{1,3}), Split([\u4e00-\u9fa5\u3040-\u309f
//!                  \u30a0-\u30ff]+), Split(word/punctuation), Digits(
//!                  individual_digits), ByteLevel(add_prefix_space=false)
//!   model          BPE, no byte_fallback, no fuse_unk, ignore_merges false
//!   post_processor ByteLevel (adds no BOS and no EOS)
//!
//! and 113 added tokens. HF matches every added token atomically before
//! pre-tokenization, whether or not it is flagged special; GGUF writes only
//! the special ones as control (type 3). The shared control split registers
//! those, and `specials` registers the 22 marked special=false by ID so a
//! rendered turn still encodes as one piece per role marker.
use super::{bpe_emit_piece, GgufFile, TokError, Vocab};
use regex::Regex;
use std::sync::OnceLock;

/// tokenizer.json and tokenizer.ggml.* of the pinned source.
const VOCAB: usize = 131_072;
const MERGES: usize = 130_716;
const PRE: &[u8] = b"spark2_5";

/// Added tokens the source marks special=false (token-check.json,
/// `chat.added_not_special.ids`): the thinking pair, `<unk>`, the three plain
/// role words, three FIM tokens and the thirteen control markers.
const ADDED_NOT_SPECIAL: [usize; 22] = [
    3, 4, 5, 8, 9, 10, 11, 12, 13, 130_972, 130_973, 130_974, 130_975, 130_976, 130_977, 130_978,
    130_979, 130_980, 130_981, 130_982, 130_983, 130_984,
];

/// Added tokens the source marks special=true; GGUF writes them as control.
const SPECIAL_ADDED: [usize; 5] = [0, 1, 2, 6, 7];
const RESERVED: std::ops::RangeInclusive<usize> = 130_986..=131_071;

/// Every marker the template emits or the protocol names, with the id the
/// pinned checkpoint resolves it to. The identity, not the string, is the
/// contract: `<｜end▁of▁sentence｜>` is eos (1) while `<｜end▁of▁text｜>` (7)
/// is a different token the template never emits.
const MARKERS: [(&str, i32); 27] = [
    ("<｜start▁of▁sentence｜>", 0),
    ("<｜end▁of▁sentence｜>", 1),
    ("<｜▁pad▁｜>", 2),
    ("<think>", 3),
    ("</think>", 4),
    ("<unk>", 5),
    ("<｜start▁of▁text｜>", 6),
    ("<｜end▁of▁text｜>", 7),
    ("<User>", 8),
    ("<System>", 9),
    ("<Bot>", 10),
    ("<｜fim▁begin｜>", 11),
    ("<｜fim▁hole｜>", 12),
    ("<｜fim▁end｜>", 13),
    ("<|System|>", 130_972),
    ("<|User|>", 130_973),
    ("<|Tool|>", 130_974),
    ("<|Developer|>", 130_975),
    ("<|Bot|>", 130_976),
    ("<tool_call>", 130_977),
    ("<tool_response>", 130_978),
    ("</tool_response>", 130_979),
    ("<arg_key>", 130_980),
    ("</arg_key>", 130_981),
    ("<arg_value>", 130_982),
    ("</arg_value>", 130_983),
    ("</tool_call>", 130_984),
];

fn invalid() -> TokError {
    TokError::InvalidTokenizer("Spark-X2.6 vocabulary/pretokenizer")
}

pub(super) fn specials(v: &mut Vocab, g: &GgufFile) -> Result<(), TokError> {
    if g.get_string("tokenizer.ggml.pre") != Some(PRE)
        || v.tokens.len() != VOCAB
        || v.merges.len() != MERGES
    {
        return Err(invalid());
    }
    verify_types(g)?;
    for &(text, id) in &MARKERS {
        if v.lookup(text)? != id {
            return Err(TokError::InvalidTokenizer("Spark-X2.6 control token IDs"));
        }
    }
    for (n, token) in RESERVED.clone().enumerate() {
        let want = format!("<||unused{}||>", n + 2);
        if v.tokens[token] != want.as_bytes() {
            return Err(TokError::InvalidTokenizer("Spark-X2.6 reserved ids"));
        }
    }

    // `<｜end▁of▁sentence｜>` is both the turn close and the stop token;
    // `<｜end▁of▁text｜>` never appears in this template.
    v.bos_id = 0;
    v.eos_id = 1;
    v.eot_id = 1;
    v.end_of_turn_id = 1;
    v.system_id = 130_972;
    v.user_id = 130_973;
    v.tool_id = 130_974;
    v.assistant_id = 130_976;
    v.think_start_id = 3;
    v.think_end_id = 4;
    v.tool_call_start_id = 130_977;
    v.tool_call_end_id = 130_984;
    v.tool_response_start_id = 130_978;
    v.tool_response_end_id = 130_979;
    v.arg_key_start_id = 130_980;
    v.arg_key_end_id = 130_981;
    v.arg_value_start_id = 130_982;
    v.arg_value_end_id = 130_983;
    // No ChatML envelope and no native writer: both stay absent.
    v.im_start_id = -1;
    v.im_end_id = -1;
    v.im_content_id = -1;
    v.dsml_id = -1;
    v.image_id = -1;

    for &id in &ADDED_NOT_SPECIAL {
        register_added(v, id);
    }
    Ok(())
}

/// The type-3 set must be exactly the special added tokens: a control token
/// outside it would be matched atomically where HF applies the merge table,
/// and a special added token inside the vocabulary range would be split where
/// HF matches it whole.
fn verify_types(g: &GgufFile) -> Result<(), TokError> {
    let array = g
        .get_array("tokenizer.ggml.token_type")
        .ok_or_else(invalid)?;
    let types = g.array_le_u32s(&array).map_err(|_| invalid())?;
    if types.len() != VOCAB {
        return Err(invalid());
    }
    let mut control = 0usize;
    for (id, &typ) in types.iter().enumerate() {
        let special = id < 14 && SPECIAL_ADDED.contains(&id) || RESERVED.contains(&id);
        if typ == 3 {
            control += 1;
            if !special {
                return Err(invalid());
            }
        } else if special {
            return Err(invalid());
        }
    }
    if control != SPECIAL_ADDED.len() + RESERVED.clone().count() {
        return Err(invalid());
    }
    Ok(())
}

/// Register one added token for the shared longest-match scan.
fn register_added(v: &mut Vocab, id: usize) {
    let token = v.tokens[id].clone();
    if token.is_empty() {
        return;
    }
    v.user_defined_max_len = v.user_defined_max_len.max(token.len() as u32);
    v.user_defined_first[token[0] as usize] = true;
    v.user_defined.insert(token, id as i32);
}

/// `Split(pattern).Isolated`: every match is its own piece and every gap
/// between matches is a piece, in order, as `NormalizedString::split` builds
/// them for tokenizers' Split pre-tokenizer.
fn split_isolated<'a>(text: &'a str, re: &Regex) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut last = 0usize;
    for m in re.find_iter(text) {
        if m.start() > last {
            out.push(&text[last..m.start()]);
        }
        out.push(&text[m.start()..m.end()]);
        last = m.end();
    }
    if last < text.len() {
        out.push(&text[last..]);
    }
    out
}

/// The three Split patterns as `find_iter` sees them. The whitespace
/// alternatives of the word pattern carry a `(?!\S)` that the `regex` crate
/// cannot compile, so that pattern is matched by `word_split` below.
fn numbers() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\p{N}{1,3}").expect("Spark digit split"))
}

fn cjk() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+")
            .expect("Spark CJK split")
    })
}

fn digit() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\p{N}").expect("Spark individual digit"))
}

fn punct_word() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r##"^(?:[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+)"##)
            .expect("Spark punctuation split")
    })
}

fn letter_word() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^(?:[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+)").expect("Spark letter")
    })
}

fn punct_run() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?: ?[\p{P}\p{S}]+)").expect("Spark punctuation"))
}

fn newline() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[\r\n]").expect("Spark newline"))
}

/// Byte length of the match at the start of `text`, if any.
fn match_len(re: &Regex, text: &str) -> usize {
    re.find(text)
        .filter(|m| m.start() == 0)
        .map_or(0, |m| m.end())
}

/// The third Split stage:
///
///   [punct][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
///   |  ?[\p{P}\p{S}]+ | [\r\n] | \s+(?!\S) | \s+
///
/// Alternatives are tried in order at each position, the way the pattern's
/// leftmost-first match does. `\s+(?!\S)` keeps the last whitespace of a run
/// out so the next alternative can attach it to a word (` word`); a run that
/// ends the piece has nothing after it and stays whole.
fn push_word_split<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let mut pos = 0usize;
    let mut start = 0usize;
    while pos < text.len() {
        let rest = &text[pos..];
        let mut len = match_len(punct_word(), rest);
        if len == 0 {
            len = match_len(letter_word(), rest);
        }
        if len == 0 {
            len = match_len(punct_run(), rest);
        }
        if len == 0 {
            len = match_len(newline(), rest);
        }
        if len == 0 {
            len = whitespace_len(rest);
        }
        if len == 0 {
            pos += rest.chars().next().map_or(1, char::len_utf8);
            continue;
        }
        if pos > start {
            out.push(&text[start..pos]);
        }
        out.push(&text[pos..pos + len]);
        pos += len;
        start = pos;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
}

/// The last two alternatives of the word pattern. The match is the whole
/// run at the end of the piece, otherwise the run without its last
/// whitespace character (`\s+(?!\S)` backtracking). A one-character run
/// cannot drop its last character, so `\s+` matches it whole.
fn whitespace_len(text: &str) -> usize {
    let mut end = 0usize;
    for (at, c) in text.char_indices() {
        if !c.is_whitespace() {
            break;
        }
        end = at + c.len_utf8();
    }
    if end == 0 {
        return 0;
    }
    if end == text.len() {
        return end;
    }
    let last = text[..end].chars().next_back().map_or(0, char::len_utf8);
    if end > last {
        end - last
    } else {
        end
    }
}

pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    let Ok(text) = std::str::from_utf8(text) else {
        // A tokenizer.json pipeline only sees Unicode; an unpaired byte has
        // no class to split on, so it is one ByteLevel piece.
        bpe_emit_piece(v, text, out);
        return;
    };

    // The four Split/Digits stages in declaration order, each subdividing the
    // pieces the previous one produced, which is what a tokenizers Sequence
    // does.
    let mut cjk_pieces = Vec::new();
    for piece in split_isolated(text, numbers()) {
        cjk_pieces.extend(split_isolated(piece, cjk()));
    }
    let mut words = Vec::with_capacity(cjk_pieces.len());
    for piece in cjk_pieces {
        push_word_split(piece, &mut words);
    }
    let mut digits = Vec::with_capacity(words.len());
    for piece in words {
        digits.extend(split_isolated(piece, digit()));
    }
    for piece in digits {
        bpe_emit_piece(v, piece.as_bytes(), out);
    }
}

#[cfg(test)]
mod tests {
    use super::{push_word_split, whitespace_len};

    fn words(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        push_word_split(text, &mut out);
        out
    }

    /// The space before a word travels with it; the rest of a run is its own
    /// piece, which is what `\s+(?!\S)` buys over a plain run.
    #[test]
    fn spaces_attach_to_the_following_word() {
        assert_eq!(words("a  b"), ["a", " ", " b"]);
        assert_eq!(words("   "), ["   "]);
        assert_eq!(words("  x"), [" ", " x"]);
    }

    /// Punctuation runs keep a leading space, ASCII words keep a leading
    /// space, and a newline is its own piece. Only one punctuation character
    /// may precede a word (`[punct][A-Za-z]+`), so a run of three is isolated
    /// before `hello`.
    #[test]
    fn punctuation_and_newlines_split() {
        assert_eq!(words("!!!hello"), ["!!!", "hello"]);
        assert_eq!(words("hi!!!"), ["hi", "!!!"]);
        assert_eq!(words("x\n y"), ["x", "\n", " y"]);
        assert_eq!(words("a/b"), ["a", "/b"]);
    }

    #[test]
    fn whitespace_run_keeps_the_last_character() {
        assert_eq!(whitespace_len("  x"), 1);
        assert_eq!(whitespace_len("   "), 3);
        assert_eq!(whitespace_len(" x"), 1);
    }
}
