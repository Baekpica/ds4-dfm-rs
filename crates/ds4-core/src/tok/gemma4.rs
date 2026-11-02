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

/// BPE over the symbol list: the lowest merge rank first, leftmost when
/// ranks tie, repeated until no adjacent pair has a rule.
fn merge(vocab: &Vocab, sym: &mut Vec<Vec<u8>>) {
    loop {
        let mut best = i32::MAX;
        for i in 0..sym.len().saturating_sub(1) {
            let rank = super::bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if rank >= 0 && rank < best {
                best = rank;
            }
        }
        if best == i32::MAX {
            return;
        }
        let mut write = 0usize;
        let mut read = 0usize;
        while read < sym.len() {
            if read + 1 < sym.len() && super::bpe_rank(vocab, &sym[read], &sym[read + 1]) == best {
                let mut merged = Vec::with_capacity(sym[read].len() + sym[read + 1].len());
                merged.extend_from_slice(&sym[read]);
                merged.extend_from_slice(&sym[read + 1]);
                sym[write] = merged;
                write += 1;
                read += 2;
            } else {
                if write != read {
                    sym[write] = std::mem::take(&mut sym[read]);
                }
                write += 1;
                read += 1;
            }
        }
        sym.truncate(write);
    }
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
