//! The checkpoint's own tokenizer, read from `tokenizer.json`.
//!
//! The artifact carries weights only, so the tokenizer is the source
//! checkpoint's file, named on the command line rather than inferred. The
//! pipeline is the one the file states, checked here instead of assumed:
//!
//! - normalizer `Replace(" " -> "▁")` (`tokenizer.json` `normalizer`);
//! - pre-tokenizer `Split(" ", MergedWithPrevious)`, which cannot fire after
//!   the normalizer has replaced every space, so the whole prompt is one BPE
//!   word;
//! - model `BPE`, `byte_fallback: true`, `fuse_unk: false`, no prefixes or
//!   suffixes;
//! - post-processor `TemplateProcessing` prepending `<bos>` (id 2).
//!
//! The merge loop mirrors `tokenizers-0.22.1`'s `Word::merge_all`
//! (`/data/refs/tokenizers/tokenizers-0.22.1/src/models/bpe/word.rs:162-220`):
//! repeatedly merge the adjacent pair with the lowest merge rank, leftmost
//! first on a tie. The initial symbols are Unicode scalar values
//! (`model.rs:382-416`), and a character outside the vocabulary falls back to
//! its bytes as `<0xXX>` tokens (`model.rs:418-433`).

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use super::Error;

const BOS: u32 = 2;

#[derive(Deserialize)]
struct File {
    model: Model,
    #[serde(default)]
    added_tokens: Vec<Added>,
}

#[derive(Deserialize)]
struct Model {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    merges: Vec<(String, String)>,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    fuse_unk: bool,
    #[serde(default)]
    unk_token: Option<String>,
}

#[derive(Deserialize)]
struct Added {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

/// The tokenizer the checkpoint ships.
pub struct Tokenizer {
    tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    /// `(a, b) -> (rank, merged)` over token ids.
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_tokens: [Option<u32>; 256],
    unk: Option<u32>,
    fuse_unk: bool,
    special: Vec<bool>,
    specials: HashMap<String, u32>,
    pub bos: u32,
    pub eos: Vec<u32>,
}

impl Tokenizer {
    /// Read `tokenizer.json`.
    pub fn open(path: &Path) -> Result<Tokenizer, Error> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        let file: File = serde_json::from_slice(&bytes)
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        if file.model.kind != "BPE" || !file.model.byte_fallback {
            return Err(Error::new(format!(
                "{}: expected a BPE model with byte fallback",
                path.display()
            )));
        }
        let mut tokens = vec![String::new(); file.model.vocab.len()];
        for (text, &id) in &file.model.vocab {
            let slot = tokens
                .get_mut(id as usize)
                .ok_or_else(|| Error::new(format!("{}: id {id} outside the table", path.display())))?;
            *slot = text.clone();
        }
        let mut merges = HashMap::with_capacity(file.model.merges.len());
        for (rank, (a, b)) in file.model.merges.into_iter().enumerate() {
            let a_id = *file.model.vocab.get(&a).ok_or_else(|| {
                Error::new(format!("{}: merge {a:?} is not in the vocabulary", path.display()))
            })?;
            let b_id = *file.model.vocab.get(&b).ok_or_else(|| {
                Error::new(format!("{}: merge {b:?} is not in the vocabulary", path.display()))
            })?;
            let merged = a.clone() + &b;
            let new_id = *file.model.vocab.get(&merged).ok_or_else(|| {
                Error::new(format!(
                    "{}: merge {merged:?} is not in the vocabulary",
                    path.display()
                ))
            })?;
            merges.insert((a_id, b_id), (rank as u32, new_id));
        }
        let mut byte_tokens = [None; 256];
        for (byte, slot) in byte_tokens.iter_mut().enumerate() {
            let code = format!("<{byte:#04X}>");
            *slot = file.model.vocab.get(&code).copied();
        }
        let mut special = vec![false; tokens.len()];
        let mut specials = HashMap::new();
        let mut eos = Vec::new();
        for added in &file.added_tokens {
            if let Some(slot) = special.get_mut(added.id as usize) {
                *slot = added.special;
            }
            if added.special {
                specials.insert(added.content.clone(), added.id);
            }
            match added.content.as_str() {
                "<eos>" | "<turn|>" => eos.push(added.id),
                _ => {}
            }
        }
        let unk = file
            .model
            .unk_token
            .as_ref()
            .and_then(|name| file.model.vocab.get(name).copied());
        Ok(Tokenizer {
            tokens,
            vocab: file.model.vocab,
            merges,
            byte_tokens,
            unk,
            fuse_unk: file.model.fuse_unk,
            special,
            specials,
            bos: BOS,
            eos,
        })
    }

    /// A special token's id by its text, e.g. `"<|turn>"`.
    pub fn special(&self, name: &str) -> Result<u32, Error> {
        self.specials
            .get(name)
            .copied()
            .ok_or_else(|| Error::new(format!("tokenizer has no {name}")))
    }

    /// Render messages with the Gemma-4 template shipped in
    /// `/data/llama.cpp/models/templates/google-gemma-4-31B-it.jinja`.
    ///
    /// The rules that matter here, read from that file:
    /// - `<bos>` opens the prompt;
    /// - a `system`/`developer` first message, or thinking enabled, opens a
    ///   `<|turn>system` block; `enable_thinking` inserts `<|think|>`;
    /// - `assistant` renders as the `model` role; content is trimmed;
    /// - every message closes with `<turn|>` and a newline;
    /// - the generation prompt is `<|turn>model\n`, and with thinking
    ///   disabled it closes an empty thought channel
    ///   (`<|channel>thought\n<channel|>`) so the reply starts as content.
    pub fn chat_prompt(
        &self,
        messages: &[(String, String)],
        thinking: bool,
    ) -> Result<Vec<u32>, Error> {
        let turn = self.special("<|turn>")?;
        let end_turn = self.special("<turn|>")?;
        let channel = self.special("<|channel>")?;
        let end_channel = self.special("<channel|>")?;
        let think = self.special("<|think|>")?;

        let mut ids = vec![self.bos];
        let mut rest = messages;
        let first_system = messages
            .first()
            .map(|(role, _)| role == "system" || role == "developer")
            .unwrap_or(false);
        if thinking || first_system {
            ids.push(turn);
            ids.extend(self.encode("system\n"));
            if thinking {
                ids.push(think);
                ids.extend(self.encode("\n"));
            }
            if first_system {
                ids.extend(self.encode(messages[0].1.trim()));
                rest = &messages[1..];
            }
            ids.push(end_turn);
            ids.extend(self.encode("\n"));
        }
        for (role, content) in rest {
            let role = if role == "assistant" { "model" } else { role.as_str() };
            ids.push(turn);
            ids.extend(self.encode(&format!("{role}\n")));
            let content = if role == "model" {
                strip_thinking(content)
            } else {
                content.trim().to_string()
            };
            ids.extend(self.encode(&content));
            ids.push(end_turn);
            ids.extend(self.encode("\n"));
        }
        ids.push(turn);
        ids.extend(self.encode("model\n"));
        if !thinking {
            ids.push(channel);
            ids.extend(self.encode("thought\n"));
            ids.push(end_channel);
        }
        Ok(ids)
    }

    /// Does this id mark an end of turn?
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }

    /// Encode one piece of text: the normalizer, the character split with
    /// byte fallback, then the merge loop. No special tokens are added.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = text.replace(' ', "\u{2581}");
        let mut symbols: Vec<u32> = Vec::with_capacity(normalized.len());
        for ch in normalized.chars() {
            let mut buf = [0u8; 4];
            let piece = ch.encode_utf8(&mut buf);
            if let Some(&id) = self.vocab.get(piece) {
                symbols.push(id);
                continue;
            }
            let mut fallback = true;
            for byte in piece.as_bytes() {
                match self.byte_tokens[*byte as usize] {
                    Some(id) => symbols.push(id),
                    None => {
                        fallback = false;
                        break;
                    }
                }
            }
            if !fallback {
                let unk = self.unk.unwrap_or(3);
                if !self.fuse_unk || symbols.last() != Some(&unk) {
                    symbols.push(unk);
                }
            }
        }
        self.merge(symbols)
    }

    /// Encode a prompt as the post-processor does: `<bos>` first.
    pub fn encode_prompt(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![self.bos];
        ids.extend(self.encode(text));
        ids
    }

    fn merge(&self, mut symbols: Vec<u32>) -> Vec<u32> {
        while symbols.len() > 1 {
            let mut best: Option<(usize, u32, u32)> = None;
            for i in 0..symbols.len() - 1 {
                if let Some(&(rank, merged)) = self.merges.get(&(symbols[i], symbols[i + 1])) {
                    if best.map(|(_, r, _)| rank < r).unwrap_or(true) {
                        best = Some((i, rank, merged));
                    }
                }
            }
            match best {
                Some((at, _, merged)) => {
                    symbols[at] = merged;
                    symbols.remove(at + 1);
                }
                None => break,
            }
        }
        symbols
    }

    /// One id's text, or `None` for a special token (skipped when decoding).
    pub fn token(&self, id: u32) -> Option<&str> {
        if self.special.get(id as usize).copied().unwrap_or(true) {
            return None;
        }
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Decode ids to text: skip specials, turn `<0xXX>` runs back into bytes,
    /// and map `▁` to a space (`tokenizer.json` `decoder`).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        let flush = |out: &mut String, bytes: &mut Vec<u8>| {
            if !bytes.is_empty() {
                out.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        };
        for &id in ids {
            let Some(text) = self.token(id) else { continue };
            if let Some(byte) = byte_of(text) {
                bytes.push(byte);
                continue;
            }
            flush(&mut out, &mut bytes);
            out.push_str(text);
        }
        flush(&mut out, &mut bytes);
        out.replace('\u{2581}', " ")
    }
}

/// `<0xXX>` -> the byte, the form the byte fallback uses
/// (`tokenizers .../model.rs:422`, `format!("<{b:#04X}>")`).
fn byte_of(token: &str) -> Option<u8> {
    let rest = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if rest.len() != 2 {
        return None;
    }
    u8::from_str_radix(rest, 16).ok()
}

/// Incremental decoding for a streaming reply.
///
/// Byte-fallback tokens carry UTF-8 one byte at a time, so a reply cannot be
/// decoded one token at a time with [`Tokenizer::decode`]: a multi-byte
/// character would be split across emissions. This holds the pending bytes
/// and only releases the valid UTF-8 prefix.
#[derive(Default)]
pub struct Decoder {
    bytes: Vec<u8>,
}

impl Decoder {
    /// Append one token's text to `out`; returns whether anything was written.
    pub fn push(&mut self, tok: &Tokenizer, id: u32, out: &mut String) -> bool {
        let Some(text) = tok.token(id) else { return false };
        if let Some(byte) = byte_of(text) {
            self.bytes.push(byte);
            return self.drain(out);
        }
        let mut wrote = self.drain(out);
        if !self.bytes.is_empty() {
            out.push_str(&String::from_utf8_lossy(&self.bytes));
            self.bytes.clear();
            wrote = true;
        }
        out.push_str(&text.replace('\u{2581}', " "));
        wrote
    }

    /// Release whatever is left.
    pub fn finish(&mut self, out: &mut String) {
        if !self.bytes.is_empty() {
            out.push_str(&String::from_utf8_lossy(&self.bytes));
            self.bytes.clear();
        }
    }

    fn drain(&mut self, out: &mut String) -> bool {
        let mut wrote = false;
        loop {
            match std::str::from_utf8(&self.bytes) {
                Ok(_) => break,
                Err(e) if e.valid_up_to() == 0 => {
                    if let Some(bad) = e.error_len() {
                        // An invalid byte can never become valid; release it
                        // as the replacement character rather than stall.
                        out.push('\u{fffd}');
                        self.bytes.drain(..bad);
                        wrote = true;
                        continue;
                    }
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    let text = std::str::from_utf8(&self.bytes[..valid]).expect("validated by from_utf8");
                    out.push_str(&text.replace('\u{2581}', " "));
                    self.bytes.drain(..valid);
                    wrote = true;
                }
            }
        }
        if !self.bytes.is_empty() {
            // Nothing more can complete now; keep the prefix if it is a
            // complete character already.
            if std::str::from_utf8(&self.bytes).is_ok() {
                let text = String::from_utf8(std::mem::take(&mut self.bytes)).expect("validated above");
                out.push_str(&text.replace('\u{2581}', " "));
                wrote = true;
            }
        }
        wrote
    }
}

/// The template's `strip_thinking` macro: drop the thought part of each
/// `<channel|>`-separated segment, up to and including its `<|channel>`.
fn strip_thinking(text: &str) -> String {
    let mut out = String::new();
    for part in text.split("<channel|>") {
        match part.find("<|channel>") {
            Some(at) => out.push_str(&part[..at]),
            None => out.push_str(part),
        }
    }
    out.trim().to_string()
}
