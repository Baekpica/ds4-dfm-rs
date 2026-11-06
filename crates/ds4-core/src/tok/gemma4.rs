//! Gemma-4 keeps the SentencePiece vocabulary as a BPE table. The source
//! `tokenizer.json` (revision 4d7ae498) declares exactly:
//!
//!   normalizer     Replace(" " -> U+2581)
//!   pre_tokenizer  Split(" "), MergedWithPrevious
//!   model          BPE, byte_fallback true, fuse_unk true, no ignore_merges
//!   post_processor TemplateProcessing over Sequence A only (no BOS)
//!   decoder        Replace(U+2581 -> " "), ByteFallback, Fuse
//!
//! The normalizer removes every U+0020 before the split stage, so the split
//! has nothing left to divide and the merge table applies to the whole
//! normalized string. A character with no vocabulary entry contributes its
//! UTF-8 bytes as `<0xNN>` symbols, which may then merge like any other
//! symbol.
use super::{GgufFile, TokError, Vocab};

/// tokenizer.ggml.tokens and .merges lengths of the pinned revision.
const VOCAB: usize = 262_144;
const MERGES: usize = 514_906;
/// The SentencePiece space marker the normalizer writes for U+0020.
const MARKER: char = '\u{2581}';
/// Control ids of the pinned revision; generation_config.json's
/// eos_token_id [1, 106, 50] is three of them.
const CONTROL_IDS: [(&str, i32); 7] = [
    ("<bos>", 2),
    ("<eos>", 1),
    ("<|turn>", 105),
    ("<turn|>", 106),
    ("<|think|>", 98),
    ("<|tool_response>", 50),
    ("<tool_response|>", 51),
];

pub(super) fn specials(v: &mut Vocab, g: &GgufFile) -> Result<(), TokError> {
    if g.get_string("tokenizer.ggml.pre") != Some(b"gemma4")
        || v.tokens.len() != VOCAB
        || v.merges.len() != MERGES
    {
        return Err(TokError::InvalidTokenizer(
            "Gemma-4 vocabulary/pretokenizer",
        ));
    }
    v.bos_id = v.lookup("<bos>")?;
    v.eos_id = v.lookup("<eos>")?;
    // `<turn|>` closes an assistant turn and `<|tool_response>` starts the
    // tool result the client sends back; generation_config.json
    // eos_token_id is [1, 106, 50], all three of these.
    v.eot_id = v.lookup("<turn|>")?;
    v.end_of_turn_id = v.eot_id;
    v.im_start_id = v.lookup("<|turn>")?;
    v.im_end_id = v.eot_id;
    // The template opens reasoning with `<|channel>thought\n` and closes it
    // with `<channel|>`; `<|think|>` is the turn-level enable control.
    v.think_start_id = v.lookup("<|think|>")?;
    v.think_end_id = v.lookup("<channel|>")?;
    v.tool_call_start_id = v.lookup("<|tool_call>")?;
    v.tool_call_end_id = v.lookup("<tool_call|>")?;
    v.tool_response_start_id = v.lookup("<|tool_response>")?;
    v.tool_response_end_id = v.lookup("<tool_response|>")?;
    v.dsml_id = -1;
    for (name, id) in CONTROL_IDS {
        if v.lookup(name)? != id {
            return Err(TokError::InvalidTokenizer("Gemma-4 control token IDs"));
        }
    }
    Ok(())
}

/// One piece through the tokenizer.json decoder, Replace(U+2581 -> " "),
/// ByteFallback, Fuse: `<0xNN>` is the byte NN and any other piece keeps its
/// own UTF-8 bytes. There is no ByteLevel stage: id 41843 is "▁però" with
/// U+00F2 in UTF-8, and GPT-2's byte map would print 0xF2, a byte the prompt
/// never had, so a re-render would stop matching the prompt.
pub(super) fn piece_text(s: &[u8]) -> Vec<u8> {
    if let Some(byte) = fallback_byte(s) {
        return vec![byte];
    }
    let mut out = Vec::with_capacity(s.len());
    let mut pos = 0usize;
    while pos < s.len() {
        let at = pos;
        if super::utf8_decode_one(s, &mut pos) == MARKER as u32 {
            out.push(b' ');
        } else {
            out.extend_from_slice(&s[at..pos]);
        }
    }
    out
}

/// The byte a `<0xNN>` token stands for.
fn fallback_byte(s: &[u8]) -> Option<u8> {
    let hex = s.strip_prefix(b"<0x")?.strip_suffix(b">")?;
    if hex.len() != 2 || !hex.iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok()
}

/// `<0xNN>` for one byte, the tokenizer.json byte fallback spelling.
fn byte_token(b: u8) -> [u8; 6] {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    [
        b'<',
        b'0',
        b'x',
        HEX[(b >> 4) as usize],
        HEX[(b & 0xf) as usize],
        b'>',
    ]
}

/// One symbol of the merge table: the token bytes, or the byte-fallback
/// tokens a character outside the vocabulary decomposes into.
fn symbols(vocab: &Vocab, text: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < text.len() {
        let mut n = super::utf8_len_from_first_byte(text[pos]);
        if pos + n > text.len() {
            n = 1;
        }
        let ch = &text[pos..pos + n];
        pos += n;
        if vocab.token_to_id.contains_key(ch) {
            out.push(ch.to_vec());
        } else {
            for &b in ch {
                out.push(byte_token(b).to_vec());
            }
        }
    }
    out
}

/// A heap entry: merge rank, left index, right index, and the stamp of each
/// end at the time the pair was ranked.
type PairEntry = (i32, u32, u32, u32, u32);

/// The symbol list as links over the original indices. A symbol keeps its
/// index when it absorbs its right neighbour, so the pair on either side of
/// it is a different pair under the same two indices; its stamp tells them
/// apart.
struct Chain {
    next: Vec<u32>,
    prev: Vec<u32>,
    live: Vec<bool>,
    stamp: Vec<u32>,
}

impl Chain {
    /// Whether the entry still names two adjacent live symbols with the
    /// bytes it was ranked on.
    fn holds(&self, &(_, i, j, si, sj): &PairEntry) -> bool {
        let (a, b) = (i as usize, j as usize);
        b < self.live.len()
            && self.live[a]
            && self.live[b]
            && self.next[a] == j
            && self.stamp[a] == si
            && self.stamp[b] == sj
    }
}

/// BPE over the symbol list, as tokenizers' `Word::merge_all` runs it:
/// merge the adjacent pair with the lowest rank, leftmost on a tie, and rank
/// the two pairs the merge creates at once, until no pair has a rule. A
/// min-heap of `PairEntry` replaces a rescan per merge, which was quadratic:
/// 34.5 s of a 35.0 s TTFT at 61,871 prompt tokens (GB10, 2026-11-05).
fn merge_by(sym: &mut Vec<Vec<u8>>, rank_of: impl Fn(&[u8], &[u8]) -> i32) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = sym.len();
    if n < 2 {
        return;
    }
    let n32 = n as u32;
    let mut chain = Chain {
        next: (1..=n32).collect(),
        prev: (0..n32).map(|i| i.wrapping_sub(1)).collect(),
        live: vec![true; n],
        stamp: vec![0; n],
    };
    let entry = |sym: &[Vec<u8>], chain: &Chain, i: u32, j: u32| {
        let rank = rank_of(&sym[i as usize], &sym[j as usize]);
        let stamps = (chain.stamp[i as usize], chain.stamp[j as usize]);
        (rank >= 0).then_some(Reverse((rank, i, j, stamps.0, stamps.1)))
    };
    let mut heap: BinaryHeap<Reverse<PairEntry>> = (0..n32 - 1)
        .filter_map(|i| entry(sym, &chain, i, i + 1))
        .collect();

    while let Some(Reverse(top)) = heap.pop() {
        if !chain.holds(&top) {
            continue;
        }
        let (_, i, j, _, _) = top;
        let (a, b) = (i as usize, j as usize);
        let right = std::mem::take(&mut sym[b]);
        sym[a].extend_from_slice(&right);
        chain.stamp[a] = chain.stamp[a].wrapping_add(1);
        chain.live[b] = false;

        // The merged symbol pairs with both neighbours.
        let after = chain.next[b];
        chain.next[a] = after;
        if after < n32 {
            chain.prev[after as usize] = i;
            heap.extend(entry(sym, &chain, i, after));
        }
        let before = chain.prev[a];
        if before < n32 {
            heap.extend(entry(sym, &chain, before, i));
        }
    }

    // Index 0 only ever absorbs, so the live chain starts there.
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        out.push(std::mem::take(&mut sym[i]));
        i = chain.next[i] as usize;
    }
    *sym = out;
}

pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    // normalizer: Replace(" ", U+2581). Every space becomes three bytes, so
    // the byte stream cannot be built in place.
    let mut marker = [0u8; 4];
    let marker = MARKER.encode_utf8(&mut marker).as_bytes();
    let mut normalized = Vec::with_capacity(text.len() + text.len() / 8);
    for &b in text {
        if b == b' ' {
            normalized.extend_from_slice(marker);
        } else {
            normalized.push(b);
        }
    }
    let mut sym = symbols(v, &normalized);
    merge_by(&mut sym, |a, b| super::bpe_rank(v, a, b));
    for s in &sym {
        if let Some(&id) = v.token_to_id.get(s) {
            out.push(id);
        } else {
            for &b in s {
                if let Some(&id) = v.token_to_id.get(&byte_token(b)[..]) {
                    out.push(id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{merge_by, piece_text};

    /// The decoder prints a piece's own UTF-8: "▁però" (id 41843) is " però",
    /// not the GPT-2 byte 0xF2 for U+00F2.
    #[test]
    fn pieces_keep_their_utf8() {
        assert_eq!(piece_text("▁però".as_bytes()), " però".as_bytes());
        assert_eq!(piece_text("日本".as_bytes()), "日本".as_bytes());
        assert_eq!(piece_text("\n\n  x".as_bytes()), b"\n\n  x");
    }

    /// ByteFallback: `<0xNN>` is one byte; other angle-bracket pieces stay.
    #[test]
    fn byte_tokens_are_their_byte() {
        assert_eq!(piece_text(b"<0x0A>"), [0x0A]);
        assert_eq!(piece_text(b"<0xF2>"), [0xF2]);
        assert_eq!(piece_text(b"<0x+F>"), b"<0x+F>");
        assert_eq!(piece_text(b"<|turn>"), b"<|turn>");
    }

    /// One character per symbol, ranked by position in `merges`, the way the
    /// table stores them.
    fn run(merges: &[(&str, &str)], text: &str) -> Vec<String> {
        let mut sym: Vec<Vec<u8>> = text.chars().map(|c| c.to_string().into_bytes()).collect();
        merge_by(&mut sym, |a, b| {
            merges
                .iter()
                .position(|&(l, r)| l.as_bytes() == a && r.as_bytes() == b)
                .map_or(-1, |i| i as i32)
        });
        sym.into_iter()
            .map(|s| String::from_utf8(s).unwrap())
            .collect()
    }

    /// A pair a merge creates is ranked at once: ("aa", "a") outranks
    /// ("a", "a"), so "aaaa" is [aaa, a], as tokenizers' merge_all gives.
    #[test]
    fn created_pairs_merge_at_once() {
        assert_eq!(run(&[("aa", "a"), ("a", "a")], "aaaa"), ["aaa", "a"]);
    }

    /// Gemma-4's table opens with ("▁" x k, "▁") from k = 30 down, so a
    /// 32-space indent is one 31-marker piece and a marker.
    #[test]
    fn whitespace_runs_take_the_long_rules() {
        let runs: Vec<String> = (1..=30).rev().map(|k| "▁".repeat(k)).collect();
        let merges: Vec<(&str, &str)> = runs.iter().map(|r| (r.as_str(), "▁")).collect();
        assert_eq!(
            run(&merges, &"▁".repeat(32)),
            ["▁".repeat(31), "▁".to_string()]
        );
    }

    /// Equal ranks merge leftmost first; the overlapping pair is then gone.
    #[test]
    fn ties_go_leftmost() {
        assert_eq!(run(&[("a", "a")], "aaa"), ["aa", "a"]);
        assert_eq!(run(&[("a", "b")], "abab"), ["ab", "ab"]);
    }

    /// A symbol keeps its index when it absorbs its right neighbour, so the
    /// entry ranked on the old bytes must not merge: "abc" with ("b", "c")
    /// first leaves (a, b) stale and ("a", "bc") has no rule.
    #[test]
    fn stale_entries_never_merge() {
        assert_eq!(run(&[("b", "c"), ("a", "b")], "abc"), ["a", "bc"]);
    }
}
