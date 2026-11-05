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
    /* The decoder stage, verbatim: Replace(U+2581 -> " "), ByteFallback, Fuse.
     * Fuse only joins pieces, so these two flags are all the printer needs. */
    v.decode_space = 0x2581;
    v.byte_fallback = true;
    /* No ByteLevel stage, so a piece holds the character itself: id 41843 is
     * "▁però" (U+00F2 in UTF-8), not the byte-mapped "Ã²". Printing it
     * through the GPT-2 byte table would yield 0xF2, a byte the prompt never
     * had, and the re-render would stop matching the prompt bytes. */
    v.utf8_piece_text = true;
    if (
        v.bos_id,
        v.eos_id,
        v.im_start_id,
        v.eot_id,
        v.think_start_id,
    ) != (2, 1, 105, 106, 98)
        || v.tool_response_start_id != 50
        || v.tool_response_end_id != 51
    {
        return Err(TokError::InvalidTokenizer("Gemma-4 control token IDs"));
    }
    Ok(())
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

/// Whether the pair `(i, next[i])` still names two adjacent live symbols
/// with the bytes the heap entry was ranked on.
#[allow(clippy::too_many_arguments)]
fn pair_holds(
    live: &[bool],
    next: &[u32],
    stamp: &[u32],
    n: u32,
    i: u32,
    j: u32,
    si: u32,
    sj: u32,
) -> bool {
    live[i as usize]
        && j < n
        && live[j as usize]
        && next[i as usize] == j
        && stamp[i as usize] == si
        && stamp[j as usize] == sj
}

/// BPE over the symbol list: the lowest merge rank first, leftmost when
/// ranks tie, repeated until no adjacent pair has a rule.
///
/// The mechanism is the reference's (SentencePiece BPE: merge the adjacent
/// pair with the lowest rank, leftmost when ranks tie, until no pair has a
/// rule). What changed is how the lowest pair is found. The scan-and-merge
/// pass this replaces rescanned every adjacent pair once per rank it merged,
/// which is quadratic in the symbol count; the host tokenizer paid 34.5 s of
/// a 35.0 s request TTFT at 61 871 cached tokens on the GB10 (2026-11-05),
/// and 7.43 s for a 60 KB prompt. `bpe_rank` is a merge-table lookup, so the
/// pass count was the whole cost.
///
/// The order is unchanged, and the differential gate is byte-identical token
/// streams against the scan on the recorded corpus:
///
/// - a min-heap holds `(rank, left, right, stamp_left, stamp_right)` for
///   every adjacent pair;
/// - one round merges exactly one rank group, in position order, so a pair
///   the round creates is not merged until the round ends, which is what one
///   scan does;
/// - an entry is stale unless both ends are live, `next[left]` is still the
///   `right` it was built from, and both stamps still match. The stamps are
///   what the adjacency test cannot see: a symbol keeps its index when it
///   absorbs its right neighbour, so the pair on either side of it is a
///   different pair under the same two indices.
fn merge(vocab: &Vocab, sym: &mut Vec<Vec<u8>>) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = sym.len();
    if n < 2 {
        return;
    }
    let n32 = n as u32;
    let mut next: Vec<u32> = (1..=n32).collect();
    let mut prev: Vec<u32> = (0..n32).map(|i| i.wrapping_sub(1)).collect();
    let mut live = vec![true; n];
    // A symbol's bytes change when it absorbs its right neighbour, and a pair
    // keeps the same two indices through that, so adjacency alone cannot tell
    // an entry from the pair it was ranked on: the stamps do.
    let mut stamp = vec![0u32; n];
    let mut heap: BinaryHeap<Reverse<PairEntry>> = BinaryHeap::with_capacity(n);
    for i in 0..n - 1 {
        let rank = super::bpe_rank(vocab, &sym[i], &sym[i + 1]);
        if rank >= 0 {
            heap.push(Reverse((
                rank,
                i as u32,
                i as u32 + 1,
                stamp[i],
                stamp[i + 1],
            )));
        }
    }

    let mut group: Vec<u32> = Vec::new();
    let mut fresh: Vec<PairEntry> = Vec::new();
    loop {
        group.clear();
        let mut rank = -1i32;
        while let Some(Reverse((r, i, j, si, sj))) = heap.pop() {
            if !pair_holds(&live, &next, &stamp, n32, i, j, si, sj) {
                continue;
            }
            rank = r;
            group.push(i);
            break;
        }
        if rank < 0 {
            break;
        }
        while let Some(&Reverse((r, i, j, si, sj))) = heap.peek() {
            if r != rank {
                break;
            }
            heap.pop();
            if pair_holds(&live, &next, &stamp, n32, i, j, si, sj) {
                group.push(i);
            }
        }
        fresh.clear();
        for &i in &group {
            let a = i as usize;
            let b = next[a] as usize;
            if b >= n || !live[a] || !live[b] {
                continue;
            }
            let mut merged = Vec::with_capacity(sym[a].len() + sym[b].len());
            merged.extend_from_slice(&sym[a]);
            merged.extend_from_slice(&sym[b]);
            sym[a] = merged;
            stamp[a] = stamp[a].wrapping_add(1);
            live[b] = false;
            let after = next[b];
            next[a] = after;
            if (after as usize) < n {
                prev[after as usize] = i;
            }
            let before = prev[a];
            if (before as usize) < n {
                let r = super::bpe_rank(vocab, &sym[before as usize], &sym[a]);
                if r >= 0 {
                    fresh.push((r, before, i, stamp[before as usize], stamp[a]));
                }
            }
            if (after as usize) < n {
                let r = super::bpe_rank(vocab, &sym[a], &sym[after as usize]);
                if r >= 0 {
                    fresh.push((r, i, after, stamp[a], stamp[after as usize]));
                }
            }
        }
        for entry in fresh.drain(..) {
            heap.push(Reverse(entry));
        }
    }

    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    loop {
        out.push(std::mem::take(&mut sym[i]));
        let j = next[i] as usize;
        if j >= n {
            break;
        }
        i = j;
    }
    *sym = out;
}

pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    // normalizer: Replace(" ", U+2581). Every space becomes three bytes, so
    // the byte stream cannot be built in place.
    let mut normalized = Vec::with_capacity(text.len() + text.len() / 8);
    for &b in text {
        if b == b' ' {
            normalized.extend_from_slice("\u{2581}".as_bytes());
        } else {
            normalized.push(b);
        }
    }
    let mut sym = symbols(v, &normalized);
    merge(v, &mut sym);
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
