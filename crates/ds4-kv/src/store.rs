//! Directory-backed KVC catalog: refresh, prefix lookup, LCP, evict.

use crate::format::{
    fill_header, is_automatic_exact_replay, is_bank_replay_v1, path_for_sha, read_envelope,
    read_header_text, read_metadata, read_path, read_text_prefix, sha_hex_name, stage_stream,
    text_sha_hex, write_path, Envelope, FormatError, Header, Record, EXT_IMAGE_PIXELS_V2,
    EXT_TOOL_MAP,
};
use crate::policy::{
    eviction_score, file_size_bytes, file_size_fits, EvictionContext, Options, ScoreEntry,
    DEFAULT_MB,
};
use std::fs;
use std::io::{self, Seek as _, SeekFrom, Write as _};
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static PAYLOAD_TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Whether a record must leave something to prefill. The bank lane admits
/// only records it can extend; the serial lane may replay an exact one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Suffix {
    Optional,
    Required,
}

/// What a search would do with the best record it can see for a key. The
/// store answers what it holds; the lane turns that into a miss reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixAnswer {
    /// Nothing is keyed by this prompt.
    None,
    /// A record the search takes. Only the caller's own budget can refuse
    /// it now, and the reuse contract has no reason for that.
    Usable,
    /// A record ruled out by model, quantization or context.
    Mismatch,
    /// A record shallower than the store's current minimum — written
    /// before the minimum was raised, and skipped by every search since.
    Shallow,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub sha: String,
    pub path: PathBuf,
    pub header: Header,
    pub file_size: u64,
}

/// A file the catalog cannot use: its payload no longer holds what its
/// header describes. The name still says which text it was keyed by.
#[derive(Clone, Debug)]
struct Damaged {
    sha: String,
    path: PathBuf,
    text_bytes: u32,
}

#[derive(Debug)]
pub struct Store {
    pub dir: PathBuf,
    pub budget_bytes: u64,
    pub reject_different_quant: bool,
    pub opt: Options,
    pub continued_last_store_tokens: i32,
    entries: Vec<Entry>,
    damaged: Vec<Damaged>,
}

#[derive(Debug)]
pub struct PayloadTemp {
    path: PathBuf,
}

impl PayloadTemp {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for PayloadTemp {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl Store {
    pub fn open(
        dir: impl AsRef<Path>,
        budget_mb: u64,
        reject_different_quant: bool,
        opt: Options,
    ) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        #[cfg(unix)]
        {
            let mut builder = fs::DirBuilder::new();
            builder.recursive(true).mode(0o700).create(&dir)?;
        }
        #[cfg(not(unix))]
        fs::create_dir_all(&dir)?;
        let budget_mb = if budget_mb == 0 {
            DEFAULT_MB
        } else {
            budget_mb
        };
        let mut store = Self {
            dir,
            budget_bytes: budget_mb * 1024 * 1024,
            reject_different_quant,
            opt,
            continued_last_store_tokens: 0,
            entries: Vec::new(),
            damaged: Vec::new(),
        };
        store.evict(0, None);
        Ok(store)
    }

    pub fn payload_temp(&self) -> io::Result<PayloadTemp> {
        loop {
            let seq = PAYLOAD_TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
            let path = self
                .dir
                .join(format!(".payload.{}.{}", std::process::id(), seq));
            let mut options = fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            match options.open(&path) {
                Ok(_) => return Ok(PayloadTemp { path }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }

    pub fn refresh(&mut self) {
        self.entries.clear();
        self.damaged.clear();
        let Ok(rd) = fs::read_dir(&self.dir) else {
            return;
        };
        for ent in rd.flatten() {
            let name = ent.file_name();
            let Some(name) = name.to_str() else { continue };
            let Some(sha) = sha_hex_name(name) else {
                continue;
            };
            let path = ent.path();
            let Ok(metadata) = read_metadata(&path) else {
                // No search can use it, but it can still be the reason a
                // prompt keyed by its text finds nothing.
                if let Ok((header, _)) = read_header_text(&path, 0) {
                    self.damaged.push(Damaged {
                        sha,
                        path,
                        text_bytes: header.text_bytes,
                    });
                }
                continue;
            };
            self.entries.push(Entry {
                sha,
                path,
                header: metadata.header,
                file_size: metadata.file_size,
            });
        }
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn write(&mut self, mut record: Record) -> io::Result<PathBuf> {
        if !file_size_fits(
            self.budget_bytes,
            record.text.len() as u64,
            record.payload.len() as u64,
            record.trailer.len() as u64,
        ) {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "KVC file exceeds budget",
            ));
        }
        record.header.text_bytes = record.text.len() as u32;
        record.header.payload_bytes = record.payload.len() as u64;
        let sha = text_sha_hex(&record.text);
        let path = path_for_sha(&self.dir, &sha);
        let incoming = EvictionContext {
            text: &record.text,
            model_id: record.header.model_id,
            quant_bits: record.header.quant_bits,
            ctx_size: record.header.ctx_size,
            reject_different_quant: self.reject_different_quant,
        };
        let extra = 48
            + 4
            + record.text.len() as u64
            + record.payload.len() as u64
            + record.trailer.len() as u64;
        self.evict(extra, Some(&incoming));
        write_path(&path, &record).map_err(|e| io::Error::other(e))?;
        self.refresh();
        Ok(path)
    }

    /// Return an already-compatible KVC entry before the caller stages its
    /// opaque native payload. A nonempty trailer follows the writer fast path.
    pub fn reuse_compatible(
        &mut self,
        mut header: Header,
        text: &[u8],
        trailer: &[u8],
    ) -> io::Result<Option<PathBuf>> {
        if !trailer.is_empty() {
            header.ext_flags |= EXT_TOOL_MAP;
        }
        let sha = text_sha_hex(text);
        let path = path_for_sha(&self.dir, &sha);
        let Ok(existing) = read_envelope(&path) else {
            return Ok(None);
        };
        let compatible = existing.header.model_id == header.model_id
            && (!self.reject_different_quant || existing.header.quant_bits == header.quant_bits)
            && existing.header.ctx_size <= header.ctx_size
            && is_automatic_exact_replay(existing.header.reason, existing.header.ext_flags)
            && is_automatic_exact_replay(header.reason, header.ext_flags)
            && existing.text == text
            && text_sha_hex(&existing.text) == sha;
        if !compatible {
            return Ok(None);
        }
        if !trailer.is_empty() {
            rewrite_compatible_trailer(&path, &existing, header.ext_flags, trailer)?;
        }
        Ok(Some(path))
    }

    /// Store an opaque native payload without buffering it in Rust memory.
    /// A nonempty trailer is the frozen server tool-map suffix.
    pub fn write_payload_file(
        &mut self,
        mut header: Header,
        text: &[u8],
        payload_path: &Path,
        trailer: &[u8],
    ) -> io::Result<PathBuf> {
        if text.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "KVC text exceeds u32 length",
            ));
        }
        if !trailer.is_empty() {
            header.ext_flags |= EXT_TOOL_MAP;
        }
        if let Some(path) = self.reuse_compatible(header.clone(), text, trailer)? {
            return Ok(path);
        }
        let sha = text_sha_hex(text);
        let path = path_for_sha(&self.dir, &sha);
        if payload_path == path {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "KVC payload source aliases destination",
            ));
        }
        let mut payload = fs::File::open(payload_path)?;
        let payload_bytes = payload.metadata()?.len();
        let extra = file_size_bytes(text.len() as u64, payload_bytes, trailer.len() as u64)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "KVC file size overflows")
            })?;
        if !file_size_fits(
            self.budget_bytes,
            text.len() as u64,
            payload_bytes,
            trailer.len() as u64,
        ) {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "KVC file exceeds budget",
            ));
        }
        header.text_bytes = text.len() as u32;
        header.payload_bytes = payload_bytes;
        let incoming = EvictionContext {
            text,
            model_id: header.model_id,
            quant_bits: header.quant_bits,
            ctx_size: header.ctx_size,
            reject_different_quant: self.reject_different_quant,
        };
        let staged = stage_stream(&path, &header, text, &mut payload, payload_bytes, trailer)
            .map_err(format_io_error)?;
        self.evict_excluding(extra, Some(&incoming), Some(&path));
        if let Err(error) = fs::rename(&staged, &path) {
            let _ = fs::remove_file(staged);
            return Err(error);
        }
        self.refresh();
        Ok(path)
    }

    pub fn read(&self, path: &Path) -> io::Result<Record> {
        read_path(path).map_err(|e| io::Error::other(e))
    }

    pub fn find_text_prefix(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
    ) -> Option<usize> {
        self.find_prefix(prompt, model_id, quant_bits, ctx_size, false)
    }

    /// Reopen the selected file and repeat the C loader's header, text, and
    /// filename-hash checks before a native payload restore consumes it.
    pub fn text_prefix_candidate(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
    ) -> io::Result<Option<(PathBuf, Envelope)>> {
        self.prefix_candidate(prompt, model_id, quant_bits, ctx_size, false)
    }

    pub fn bank_text_prefix_candidate(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
    ) -> io::Result<Option<(PathBuf, Envelope)>> {
        self.prefix_candidate(prompt, model_id, quant_bits, ctx_size, true)
    }

    pub fn bank_text_prefix_candidate_identity(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        identity_flags: u8,
    ) -> io::Result<Option<(PathBuf, Envelope)>> {
        self.prefix_candidate_identity(
            prompt,
            model_id,
            quant_bits,
            ctx_size,
            true,
            EXT_IMAGE_PIXELS_V2,
            identity_flags,
        )
    }

    pub fn bank_text_lcp_candidate(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        min_lcp: usize,
    ) -> io::Result<Option<(PathBuf, Envelope, usize)>> {
        self.bank_text_lcp_candidate_with_identity(
            prompt, model_id, quant_bits, ctx_size, min_lcp, 0, 0,
        )
    }

    pub fn bank_text_lcp_candidate_identity(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        min_lcp: usize,
        identity_flags: u8,
    ) -> io::Result<Option<(PathBuf, Envelope, usize)>> {
        self.bank_text_lcp_candidate_with_identity(
            prompt,
            model_id,
            quant_bits,
            ctx_size,
            min_lcp,
            EXT_IMAGE_PIXELS_V2,
            identity_flags,
        )
    }

    fn bank_text_lcp_candidate_with_identity(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        min_lcp: usize,
        identity_mask: u8,
        identity_flags: u8,
    ) -> io::Result<Option<(PathBuf, Envelope, usize)>> {
        let Some((index, _)) = self.find_text_lcp_with_identity(
            prompt,
            model_id,
            quant_bits,
            ctx_size,
            min_lcp,
            identity_mask,
            identity_flags,
        ) else {
            return Ok(None);
        };
        let entry = self.entries[index].clone();
        let envelope = read_envelope(&entry.path).map_err(format_io_error)?;
        let header = &envelope.header;
        let unchanged = header.model_id == entry.header.model_id
            && header.quant_bits == entry.header.quant_bits
            && header.ctx_size == entry.header.ctx_size
            && header.tokens == entry.header.tokens
            && header.text_bytes == entry.header.text_bytes;
        let lcp = prompt
            .iter()
            .zip(&envelope.text)
            .take_while(|(left, right)| left == right)
            .count();
        // Corruption past the shared prefix still leaves the scan's answer
        // standing, so check the record against its own name here too.
        if !unchanged || text_sha_hex(&envelope.text) != entry.sha {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "KVC record does not match its name",
            ));
        }

        if !is_automatic_exact_replay(header.reason, header.ext_flags)
            || header.model_id != model_id
            || (self.reject_different_quant && header.quant_bits != quant_bits)
            || header.ctx_size > ctx_size
            || header.ext_flags & identity_mask != identity_flags & identity_mask
            || lcp < min_lcp
            || 8 * (lcp as u64) < u64::from(header.text_bytes)
        {
            return Ok(None);
        }
        Ok(Some((entry.path, envelope, lcp)))
    }

    fn prefix_candidate(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        require_suffix: bool,
    ) -> io::Result<Option<(PathBuf, Envelope)>> {
        self.prefix_candidate_identity(prompt, model_id, quant_bits, ctx_size, require_suffix, 0, 0)
    }

    fn prefix_candidate_identity(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        require_suffix: bool,
        identity_mask: u8,
        identity_flags: u8,
    ) -> io::Result<Option<(PathBuf, Envelope)>> {
        let index = if require_suffix {
            self.find_prefix_with_identity(
                prompt,
                model_id,
                quant_bits,
                ctx_size,
                true,
                identity_mask,
                identity_flags,
            )
        } else {
            self.find_prefix_with_identity(
                prompt,
                model_id,
                quant_bits,
                ctx_size,
                false,
                identity_mask,
                identity_flags,
            )
        };
        let Some(index) = index else {
            return Ok(None);
        };
        let entry = self.entries[index].clone();
        let envelope = read_envelope(&entry.path).map_err(format_io_error)?;
        let header = &envelope.header;
        let unchanged = header.model_id == entry.header.model_id
            && header.quant_bits == entry.header.quant_bits
            && header.ctx_size == entry.header.ctx_size
            && header.tokens == entry.header.tokens
            && header.text_bytes == entry.header.text_bytes;
        // The record no longer describes itself: its text does not hash to
        // its own name, it is not this prompt's prefix after all, or the
        // header moved under the read. The candidate was selected and then
        // refused by its own contents, which is not an absence.
        if !unchanged
            || text_sha_hex(&envelope.text) != entry.sha
            || !prompt.starts_with(&envelope.text)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "KVC record does not match its name",
            ));
        }

        // Sound, but not usable in this shape: nothing left to prefill, or
        // an identity this search does not take.
        let bank_without_logits = is_bank_replay_v1(header.reason, header.ext_flags)
            && envelope.text.len() == prompt.len();
        let missing_suffix = require_suffix && envelope.text.len() >= prompt.len();
        if !is_automatic_exact_replay(header.reason, header.ext_flags)
            || header.model_id != model_id
            || header.ext_flags & identity_mask != identity_flags & identity_mask
            || bank_without_logits
            || missing_suffix
        {
            return Ok(None);
        }
        Ok(Some((entry.path, envelope)))
    }

    pub fn touch_hit(&mut self, path: &Path) -> io::Result<()> {
        let envelope = read_envelope(path).map_err(format_io_error)?;
        let header = envelope.header;
        let raw = fill_header(
            header.model_id,
            header.quant_bits,
            header.reason,
            header.ext_flags,
            header.tokens,
            header.hits.wrapping_add(1),
            header.ctx_size,
            header.created_at,
            now_secs(),
            header.payload_bytes,
        );
        let mut file = fs::OpenOptions::new().read(true).write(true).open(path)?;
        file.write_all(&raw)?;
        file.flush()?;
        self.refresh();
        Ok(())
    }

    pub fn discard(&mut self, path: &Path) -> io::Result<()> {
        self.discard_inner(path, true)
    }

    pub fn discard_bank(&mut self, path: &Path) -> io::Result<()> {
        self.discard_inner(path, false)
    }

    fn discard_inner(&mut self, path: &Path, reset_continued: bool) -> io::Result<()> {
        if !self.entries.iter().any(|entry| entry.path == path) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "KVC discard path is not a catalog entry",
            ));
        }
        if reset_continued {
            self.continued_last_store_tokens = 0;
        }
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        self.refresh();
        Ok(())
    }

    pub fn find_bank_text_prefix(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
    ) -> Option<usize> {
        self.find_prefix(prompt, model_id, quant_bits, ctx_size, true)
    }

    fn find_prefix(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        require_suffix: bool,
    ) -> Option<usize> {
        self.find_prefix_with_identity(prompt, model_id, quant_bits, ctx_size, require_suffix, 0, 0)
    }

    fn find_prefix_with_identity(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        require_suffix: bool,
        identity_mask: u8,
        identity_flags: u8,
    ) -> Option<usize> {
        self.refresh();
        let prompt_bytes = prompt.len();
        let mut best: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !is_automatic_exact_replay(e.header.reason, e.header.ext_flags) {
                continue;
            }
            if e.header.ext_flags & identity_mask != identity_flags & identity_mask {
                continue;
            }
            if e.header.text_bytes as usize > prompt_bytes {
                continue;
            }
            if (require_suffix || is_bank_replay_v1(e.header.reason, e.header.ext_flags))
                && e.header.text_bytes as usize == prompt_bytes
            {
                continue;
            }
            if (e.header.tokens as i32) < self.opt.min_tokens {
                continue;
            }
            if e.header.model_id != model_id {
                continue;
            }
            if ctx_size < e.header.ctx_size {
                continue;
            }
            if self.reject_different_quant && e.header.quant_bits != quant_bits {
                continue;
            }
            if let Some(b) = best {
                let be = &self.entries[b];
                if (e.header.text_bytes as usize) < be.header.text_bytes as usize {
                    continue;
                }
                if e.header.text_bytes == be.header.text_bytes
                    && e.header.tokens <= be.header.tokens
                {
                    continue;
                }
            }
            let sha = text_sha_hex(&prompt[..e.header.text_bytes as usize]);
            if sha == e.sha {
                best = Some(i);
            }
        }
        best
    }

    /// What the serial search would do with this prompt: it keys on the
    /// rendered text and may replay a record of exactly that length.
    pub fn prefix_answer(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
    ) -> PrefixAnswer {
        self.prefix_answer_masked(
            prompt,
            model_id,
            quant_bits,
            ctx_size,
            Suffix::Optional,
            0,
            0,
        )
    }

    /// What the bank search would do: it keys an image request by its
    /// media-marked cache text and admits only records it can extend, so
    /// asking about the rendered prompt would see nothing at all.
    pub fn bank_prefix_answer(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        identity_flags: u8,
    ) -> PrefixAnswer {
        self.prefix_answer_masked(
            prompt,
            model_id,
            quant_bits,
            ctx_size,
            Suffix::Required,
            EXT_IMAGE_PIXELS_V2,
            identity_flags,
        )
    }

    fn prefix_answer_masked(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        suffix: Suffix,
        identity_mask: u8,
        identity_flags: u8,
    ) -> PrefixAnswer {
        self.refresh();
        let reject_quant = self.reject_different_quant;
        let min_tokens = self.opt.min_tokens;
        let mut answer = PrefixAnswer::None;

        for e in &self.entries {
            if !is_automatic_exact_replay(e.header.reason, e.header.ext_flags)
                || e.header.text_bytes as usize > prompt.len()
            {
                continue;
            }
            let exact_length = e.header.text_bytes as usize == prompt.len();
            if suffix == Suffix::Required && exact_length {
                // Nothing left to prefill from it in this shape.
                continue;
            }
            if text_sha_hex(&prompt[..e.header.text_bytes as usize]) != e.sha {
                continue;
            }

            // A bank snapshot carries no logits, so an exact replay cannot
            // use one: the record is there and its layout refuses. The media
            // flag is identity in the same way — a record keyed by this text
            // without it is a mismatch, not one about another conversation.
            let compatible = !(exact_length
                && is_bank_replay_v1(e.header.reason, e.header.ext_flags))
                && e.header.model_id == model_id
                && ctx_size >= e.header.ctx_size
                && (!reject_quant || e.header.quant_bits == quant_bits)
                && e.header.ext_flags & identity_mask == identity_flags & identity_mask;
            if compatible && (e.header.tokens as i32) >= min_tokens {
                return PrefixAnswer::Usable;
            }
            answer = weaker_answer(answer, compatible);
        }

        // A record whose payload was truncated away is not in the catalog,
        // but its name still says it was keyed by this text. It is why the
        // prompt finds nothing, and that is a mismatch.
        if self.damaged_prefix(prompt, suffix) {
            answer = weaker_answer(answer, false);
        }

        answer
    }

    /// A file the catalog dropped that still shares this prompt's required
    /// prefix. Its payload is gone; the text in front of it need not be.
    fn damaged_lcp(&self, prompt: &[u8], min_lcp: usize) -> bool {
        self.damaged.iter().any(|d| {
            if (d.text_bytes as usize) < min_lcp
                || u64::from(d.text_bytes) > 8 * prompt.len() as u64
            {
                return false;
            }
            let want = (d.text_bytes as usize).min(prompt.len());
            let Ok((_, text)) = read_header_text(&d.path, want) else {
                return false;
            };
            let lcp = text
                .iter()
                .zip(prompt)
                .take_while(|(stored, asked)| stored == asked)
                .count();
            lcp >= min_lcp && 8 * (lcp as u64) >= u64::from(d.text_bytes)
        })
    }

    /// A file the catalog dropped, keyed by a prefix of this prompt.
    fn damaged_prefix(&self, prompt: &[u8], suffix: Suffix) -> bool {
        self.damaged.iter().any(|d| {
            let text_bytes = d.text_bytes as usize;
            if text_bytes > prompt.len()
                || (suffix == Suffix::Required && text_bytes == prompt.len())
            {
                return false;
            }
            text_sha_hex(&prompt[..text_bytes]) == d.sha
        })
    }

    /// The LCP form: an edited prompt diverges from the record it shares a
    /// prefix with, so the exact question cannot see it at all. `min_lcp`
    /// is the partial minimum the bank search admits from.
    pub fn bank_lcp_answer(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        min_lcp: usize,
        identity_flags: u8,
    ) -> PrefixAnswer {
        if min_lcp == 0 || prompt.len() < min_lcp {
            return PrefixAnswer::None;
        }
        self.refresh();
        let reject_quant = self.reject_different_quant;
        let min_tokens = self.opt.min_tokens;
        let mut answer = PrefixAnswer::None;

        for e in &self.entries {
            if !is_automatic_exact_replay(e.header.reason, e.header.ext_flags)
                || (e.header.text_bytes as usize) < min_lcp
                || u64::from(e.header.text_bytes) > 8 * prompt.len() as u64
            {
                continue;
            }
            let want = (e.header.text_bytes as usize).min(prompt.len());
            let Ok((metadata, text)) = read_text_prefix(&e.path, want) else {
                continue;
            };
            if metadata.header.text_bytes != e.header.text_bytes {
                continue;
            }
            let lcp = text
                .iter()
                .zip(prompt)
                .take_while(|(stored, asked)| stored == asked)
                .count();
            if lcp < min_lcp || 8 * (lcp as u64) < u64::from(e.header.text_bytes) {
                continue;
            }

            let compatible = e.header.model_id == model_id
                && ctx_size >= e.header.ctx_size
                && (!reject_quant || e.header.quant_bits == quant_bits)
                && e.header.ext_flags & EXT_IMAGE_PIXELS_V2 == identity_flags & EXT_IMAGE_PIXELS_V2;
            if compatible && (e.header.tokens as i32) >= min_tokens {
                return PrefixAnswer::Usable;
            }
            answer = weaker_answer(answer, compatible);
        }

        // An edited prompt diverges from the record, so the damaged list has
        // to be read through the same shared prefix, not by name.
        if self.damaged_lcp(prompt, min_lcp) {
            answer = weaker_answer(answer, false);
        }

        answer
    }

    pub fn find_text_lcp(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        min_lcp: usize,
    ) -> Option<(usize, usize)> {
        self.find_text_lcp_with_identity(prompt, model_id, quant_bits, ctx_size, min_lcp, 0, 0)
    }

    fn find_text_lcp_with_identity(
        &mut self,
        prompt: &[u8],
        model_id: u8,
        quant_bits: u8,
        ctx_size: u32,
        min_lcp: usize,
        identity_mask: u8,
        identity_flags: u8,
    ) -> Option<(usize, usize)> {
        if min_lcp == 0 || prompt.len() < min_lcp {
            return None;
        }
        self.refresh();
        let mut best: Option<(usize, usize, u64)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if !is_automatic_exact_replay(e.header.reason, e.header.ext_flags) {
                continue;
            }
            if e.header.ext_flags & identity_mask != identity_flags & identity_mask {
                continue;
            }
            if (e.header.tokens as i32) < self.opt.min_tokens {
                continue;
            }
            if e.header.model_id != model_id || ctx_size < e.header.ctx_size {
                continue;
            }
            if self.reject_different_quant && e.header.quant_bits != quant_bits {
                continue;
            }
            if (e.header.text_bytes as usize) < min_lcp {
                continue;
            }
            if u64::from(e.header.text_bytes) > 8 * prompt.len() as u64 {
                continue;
            }
            let want = (e.header.text_bytes as usize).min(prompt.len());
            let Ok((metadata, text)) = read_text_prefix(&e.path, want) else {
                continue;
            };
            if metadata.header.text_bytes != e.header.text_bytes {
                continue;
            }
            let mut lcp = 0;
            while lcp < text.len() && text[lcp] == prompt[lcp] {
                lcp += 1;
            }
            if lcp < min_lcp {
                continue;
            }
            if 8 * (lcp as u64) < u64::from(e.header.text_bytes) {
                continue;
            }
            if let Some((_, best_lcp, best_text)) = best {
                if lcp < best_lcp {
                    continue;
                }
                if lcp == best_lcp && u64::from(e.header.text_bytes) >= best_text {
                    continue;
                }
            }
            best = Some((i, lcp, u64::from(e.header.text_bytes)));
        }
        best.map(|(i, lcp, _)| (i, lcp))
    }

    pub fn evict(&mut self, extra_bytes: u64, incoming: Option<&EvictionContext<'_>>) {
        self.evict_excluding(extra_bytes, incoming, None);
    }

    fn evict_excluding(
        &mut self,
        extra_bytes: u64,
        incoming: Option<&EvictionContext<'_>>,
        protected: Option<&Path>,
    ) {
        if self.budget_bytes == 0 || extra_bytes > self.budget_bytes {
            return;
        }
        self.refresh();
        let now = now_secs();
        let target = self.budget_bytes - extra_bytes;
        loop {
            let total: u64 = self
                .entries
                .iter()
                .filter(|e| protected.is_none_or(|path| path != e.path))
                .map(|e| e.file_size)
                .sum();
            if total <= target || self.entries.is_empty() {
                break;
            }
            let mut victim: Option<usize> = None;
            let mut victim_score = 0.0;
            for i in 0..self.entries.len() {
                if protected.is_some_and(|path| path == self.entries[i].path) {
                    continue;
                }
                let score = self.score_at(i, now, incoming);
                if victim.is_none()
                    || score < victim_score
                    || (score == victim_score
                        && self.entries[i].header.last_used
                            < self.entries[victim.unwrap()].header.last_used)
                {
                    victim = Some(i);
                    victim_score = score;
                }
            }
            let Some(victim) = victim else { break };
            let path = self.entries[victim].path.clone();
            let unlinked = fs::remove_file(path).is_ok();
            self.entries.remove(victim);
            if !unlinked {
                break;
            }
        }
    }

    fn score_at(&self, i: usize, now: u64, incoming: Option<&EvictionContext<'_>>) -> f64 {
        let e = &self.entries[i];
        eviction_score(
            &ScoreEntry {
                sha: &e.sha,
                quant_bits: e.header.quant_bits,
                model_id: e.header.model_id,
                reason: e.header.reason,
                tokens: e.header.tokens,
                hits: e.header.hits,
                ctx_size: e.header.ctx_size,
                created_at: e.header.created_at,
                last_used: e.header.last_used,
                text_bytes: u64::from(e.header.text_bytes),
                file_size: e.file_size,
            },
            now,
            incoming,
        )
    }
}

fn rewrite_compatible_trailer(
    path: &Path,
    existing: &Envelope,
    incoming_ext_flags: u8,
    trailer: &[u8],
) -> io::Result<()> {
    // The C fast path is O(trailer); copying a multi-GiB payload here would
    // turn a compatible save into a full checkpoint rewrite.
    let payload_end = existing
        .payload_offset
        .checked_add(existing.header.payload_bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "KVC payload end overflows"))?;
    let mut file = fs::OpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::Start(payload_end))?;
    file.set_len(payload_end)?;
    file.write_all(trailer)?;
    file.flush()?;

    let mut header = existing.header.clone();
    header.ext_flags |= incoming_ext_flags & EXT_TOOL_MAP;
    header.last_used = now_secs();
    let raw = fill_header(
        header.model_id,
        header.quant_bits,
        header.reason,
        header.ext_flags,
        header.tokens,
        header.hits,
        header.ctx_size,
        header.created_at,
        header.last_used,
        header.payload_bytes,
    );
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&raw)?;
    file.flush()
}

/// A record the search would not take: identity and layout speak before
/// depth, the way the lanes report them.
fn weaker_answer(seen: PrefixAnswer, compatible: bool) -> PrefixAnswer {
    if !compatible {
        return PrefixAnswer::Mismatch;
    }
    if seen == PrefixAnswer::None {
        return PrefixAnswer::Shallow;
    }
    seen
}

fn format_io_error(error: FormatError) -> io::Error {
    match error {
        FormatError::Io(error) => error,
        FormatError::Truncated => {
            io::Error::new(io::ErrorKind::UnexpectedEof, "truncated KVC payload")
        }
        other => io::Error::new(io::ErrorKind::InvalidData, other),
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        fill_header, le_put32, read_metadata, read_text_prefix, Header, Reason, FIXED_HEADER,
    };
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt as _;

    fn rec(text: &[u8], tokens: u32) -> Record {
        Record {
            header: Header {
                quant_bits: 2,
                reason: Reason::Cold,
                ext_flags: 0,
                model_id: 0,
                tokens,
                hits: 0,
                ctx_size: 2048,
                created_at: 1,
                last_used: 1,
                payload_bytes: 1,
                text_bytes: text.len() as u32,
            },
            text: text.to_vec(),
            payload: vec![0xAA],
            trailer: Vec::new(),
        }
    }

    #[cfg(unix)]
    #[test]
    fn new_nested_store_directories_are_private() {
        let base = std::env::temp_dir().join(format!("ds4-kv-private-dir-{}", std::process::id()));
        let dir = base.join("nested/store");
        let _ = fs::remove_dir_all(&base);

        let _store = Store::open(&dir, 16, false, Options::default()).unwrap();
        for path in [&base, &base.join("nested"), &dir] {
            assert_eq!(fs::metadata(path).unwrap().permissions().mode() & 0o077, 0);
        }

        let _ = fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn existing_store_directory_mode_is_unchanged() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-existing-dir-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        let _store = Store::open(&dir, 16, false, Options::default()).unwrap();
        assert_eq!(
            fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o755
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn payload_temp_is_private_and_drop_removes_it() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-payload-temp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store::open(&dir, 16, false, Options::default()).unwrap();

        let temp = store.payload_temp().unwrap();
        let path = temp.path().to_path_buf();
        assert!(path.exists());
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        drop(temp);
        assert!(!path.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn payload_temps_are_distinct_and_drop_independently() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-payload-pair-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let store = Store::open(&dir, 16, false, Options::default()).unwrap();

        let first = store.payload_temp().unwrap();
        let second = store.payload_temp().unwrap();
        let first_path = first.path().to_path_buf();
        let second_path = second.path().to_path_buf();
        assert_ne!(first_path, second_path);
        assert!(first_path.exists());
        assert!(second_path.exists());

        drop(first);
        assert!(!first_path.exists());
        assert!(second_path.exists());
        drop(second);
        assert!(!second_path.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lcp_candidate_refuses_a_record_that_moved() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-lcp-candidate-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut record = rec(b"shared opening/stored tail", 512);
        record.header.reason = Reason::BankShutdown;
        record.header.ext_flags = crate::format::EXT_BANK_REPLAY_V1;
        let path = store.write(record).unwrap();
        let edited = b"shared opening/edited tail";

        let (got, _, lcp) = store
            .bank_text_lcp_candidate(edited, 0, 2, 8192, 8)
            .unwrap()
            .unwrap();
        assert_eq!(got, path);
        assert_eq!(lcp, 15);

        // Corrupt the text past the shared prefix: the scan still picks the
        // record, and only the full-text check can see that it moved.
        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start((FIXED_HEADER + 4 + 20) as u64))
            .unwrap();
        file.write_all(b"X").unwrap();
        file.flush().unwrap();
        assert_eq!(
            store
                .bank_text_lcp_candidate(edited, 0, 2, 8192, 8)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_truncated_record_answers_for_the_key_it_kept() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-truncated-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let path = store.write(rec(b"shared prefix", 512)).unwrap();

        // Cut the payload away: the header and text stay, so the catalog
        // drops the record while its name still says what it was keyed by.
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(FIXED_HEADER as u64 + 4 + 13)
            .unwrap();

        assert_eq!(
            store.prefix_answer(b"shared prefix and suffix", 0, 2, 2048),
            PrefixAnswer::Mismatch
        );
        assert_eq!(
            store.prefix_answer(b"another conversation", 0, 2, 2048),
            PrefixAnswer::None
        );

        // An edited prompt diverges from the record, so only the shared
        // prefix can find it — and the text in front of the payload is
        // still there to be read.
        assert_eq!(
            store.bank_lcp_answer(b"shared prefiX edited", 0, 2, 2048, 8, 0),
            PrefixAnswer::Mismatch
        );
        assert_eq!(
            store.bank_lcp_answer(b"another opening turn", 0, 2, 2048, 8, 0),
            PrefixAnswer::None
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_exact_bank_record_answers_with_its_layout() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-exact-bank-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut record = rec(b"exact conversation", 512);
        record.header.reason = Reason::BankShutdown;
        record.header.ext_flags = crate::format::EXT_BANK_REPLAY_V1;
        store.write(record).unwrap();

        // A bank snapshot has no logits to replay from, so an exact serial
        // prompt is refused by the record's layout, not by its absence.
        assert_eq!(
            store.prefix_answer(b"exact conversation", 0, 2, 2048),
            PrefixAnswer::Mismatch
        );
        // With a suffix to prefill, the same record is what the search takes.
        assert_eq!(
            store.prefix_answer(b"exact conversation plus a turn", 0, 2, 2048),
            PrefixAnswer::Usable
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_under_the_minimum_answers_shallow() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-shallow-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        // Written while the minimum was lower; every search skips it now.
        store.write(rec(b"shared opening", 8)).unwrap();

        assert_eq!(
            store.prefix_answer(b"shared opening and more", 0, 2, 2048),
            PrefixAnswer::Shallow
        );
        assert_eq!(
            store.prefix_answer(b"another conversation", 0, 2, 2048),
            PrefixAnswer::None
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_image_record_is_a_mismatch_only_under_its_own_key() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-image-key-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, true, Options::default()).unwrap();
        let mut record = rec(b"chat\xffDS4IMG2 turn", 512);
        record.header.reason = Reason::BankCheckpoint;
        record.header.ext_flags = crate::format::EXT_BANK_REPLAY_V1 | EXT_IMAGE_PIXELS_V2;
        store.write(record).unwrap();
        let key = b"chat\xffDS4IMG2 turn and one more";

        // The identity the record was stored with: the search takes it.
        assert_eq!(
            store.bank_prefix_answer(key, 0, 2, 2048, EXT_IMAGE_PIXELS_V2),
            PrefixAnswer::Usable
        );
        // A different quantization rules it out, so the miss is a mismatch.
        assert_eq!(
            store.bank_prefix_answer(key, 0, 4, 2048, EXT_IMAGE_PIXELS_V2),
            PrefixAnswer::Mismatch
        );
        // The rendered prompt is not the key it was stored under, so the
        // text-keyed search sees nothing. Asked under this key without the
        // media flag, the record is there and its layout rules it out.
        assert_eq!(
            store.prefix_answer(b"chat turn and one more", 0, 4, 2048),
            PrefixAnswer::None
        );
        assert_eq!(
            store.bank_prefix_answer(key, 0, 2, 2048, 0),
            PrefixAnswer::Mismatch
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_edited_prompt_sees_an_identity_mismatch_through_the_lcp() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-lcp-identity-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        store
            .write(rec(b"shared opening turn/ORIGINAL", 512))
            .unwrap();
        let edited = b"shared opening turn/EDITED CONTINUATION";

        // The stored text diverges from the prompt, so the exact-prefix
        // question cannot see the record at all.
        assert_eq!(
            store.bank_prefix_answer(edited, 1, 2, 2048, 0),
            PrefixAnswer::None
        );
        // Through the LCP the partial search would have taken it, and only
        // the model it was written for rules it out.
        assert_eq!(
            store.bank_lcp_answer(edited, 1, 2, 2048, 8, 0),
            PrefixAnswer::Mismatch
        );
        // The identity it was stored with is not a mismatch.
        assert_eq!(
            store.bank_lcp_answer(edited, 0, 2, 2048, 8, 0),
            PrefixAnswer::Usable
        );
        // Neither is a prompt that shares nothing with it.
        assert_eq!(
            store.bank_lcp_answer(b"a different opening", 1, 2, 2048, 8, 0),
            PrefixAnswer::None
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discard_removes_existing_candidate_and_refreshes_index() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-discard-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let discarded = store.write(rec(b"discarded", 512)).unwrap();
        let retained = store.write(rec(b"retained", 1024)).unwrap();
        store.continued_last_store_tokens = 512;
        assert_eq!(store.entries().len(), 2);

        store.discard(&discarded).unwrap();

        assert!(!discarded.exists());
        assert!(retained.exists());
        assert_eq!(store.entries().len(), 1);
        assert_eq!(store.entries()[0].path, retained);
        assert_eq!(store.continued_last_store_tokens, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bank_discard_preserves_the_serial_checkpoint_marker() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-bank-discard-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let discarded = store.write(rec(b"bank-corrupt", 512)).unwrap();
        store.continued_last_store_tokens = 777;

        store.discard_bank(&discarded).unwrap();

        assert!(!discarded.exists());
        assert!(store.entries().is_empty());
        assert_eq!(store.continued_last_store_tokens, 777);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discard_missing_candidate_is_ok_and_refreshes_index() {
        let dir =
            std::env::temp_dir().join(format!("ds4-kv-discard-missing-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let missing = store.write(rec(b"missing", 512)).unwrap();
        store.continued_last_store_tokens = 512;
        fs::remove_file(&missing).unwrap();
        assert_eq!(store.entries().len(), 1);

        store.discard(&missing).unwrap();

        assert!(store.entries().is_empty());
        assert_eq!(store.continued_last_store_tokens, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discard_rejects_paths_outside_the_catalog() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-discard-scope-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let outside = dir.with_extension("outside");
        fs::write(&outside, b"keep").unwrap();

        let error = store.discard(&outside).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(fs::read(&outside).unwrap(), b"keep");
        fs::remove_file(outside).unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prefix_prefers_longest_matching_text() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-prefix-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        store.write(rec(b"hello", 512)).unwrap();
        store.write(rec(b"hello world", 512)).unwrap();
        let idx = store.find_text_prefix(b"hello world!", 0, 2, 2048).unwrap();
        assert_eq!(store.entries()[idx].header.text_bytes, 11);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bank_lookup_requires_suffix() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-bank-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut r = rec(b"hello", 512);
        r.header.reason = Reason::BankEvict;
        r.header.ext_flags = crate::format::EXT_BANK_REPLAY_V1;
        store.write(r).unwrap();
        assert!(store.find_bank_text_prefix(b"hello", 0, 2, 2048).is_none());
        assert!(store.find_bank_text_prefix(b"hello!", 0, 2, 2048).is_some());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bank_identity_lookup_separates_image_and_text_records() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-bank-identity-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut plain = rec(b"plain identity", 512);
        plain.header.reason = Reason::BankCheckpoint;
        plain.header.ext_flags = crate::format::EXT_BANK_REPLAY_V1;
        store.write(plain).unwrap();
        let mut image = rec(b"image identity", 512);
        image.header.reason = Reason::BankCheckpoint;
        image.header.ext_flags =
            crate::format::EXT_BANK_REPLAY_V1 | crate::format::EXT_IMAGE_PIXELS_V2;
        store.write(image).unwrap();

        assert!(store
            .bank_text_prefix_candidate_identity(
                b"plain identity tail",
                0,
                2,
                2048,
                EXT_IMAGE_PIXELS_V2,
            )
            .unwrap()
            .is_none());
        assert!(store
            .bank_text_prefix_candidate_identity(b"image identity tail", 0, 2, 2048, 0)
            .unwrap()
            .is_none());
        assert!(store
            .bank_text_prefix_candidate_identity(
                b"image identity tail",
                0,
                2,
                2048,
                EXT_IMAGE_PIXELS_V2,
            )
            .unwrap()
            .is_some());
        assert!(
            store
                .bank_text_lcp_candidate_identity(
                    b"image identitX",
                    0,
                    2,
                    2048,
                    8,
                    EXT_IMAGE_PIXELS_V2,
                )
                .unwrap()
                .is_some()
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lcp_candidate_reopens_and_revalidates_the_selected_record() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-lcp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut record = rec(b"shared alpha", 512);
        record.header.reason = Reason::BankCheckpoint;
        record.header.ext_flags = crate::format::EXT_BANK_REPLAY_V1;
        let path = store.write(record).unwrap();

        let (candidate, envelope, lcp) = store
            .bank_text_lcp_candidate(b"shared beta", 0, 2, 2048, 6)
            .unwrap()
            .unwrap();
        assert_eq!(candidate, path);
        assert_eq!(envelope.text, b"shared alpha");
        assert_eq!(lcp, b"shared ".len());

        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start((FIXED_HEADER + 4) as u64))
            .unwrap();
        file.write_all(b"X").unwrap();
        file.flush().unwrap();
        assert!(store
            .bank_text_lcp_candidate(b"shared beta", 0, 2, 2048, 6)
            .unwrap()
            .is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn indexes_sparse_payload_without_reading_it() {
        const PAYLOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;
        const TRAILER_BYTES: u64 = 17;

        let dir = std::env::temp_dir().join(format!("ds4-kv-sparse-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let text = [b'x'; 64];
        let prompt = [b'x'; 8];
        let path = path_for_sha(&dir, &text_sha_hex(&text));
        let header = fill_header(0, 2, Reason::Cold, 0, 512, 0, 2048, 1, 1, PAYLOAD_BYTES);
        let mut text_len = [0u8; 4];
        le_put32(&mut text_len, text.len() as u32);
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&text_len).unwrap();
        file.write_all(&text).unwrap();
        let payload_offset = (FIXED_HEADER + 4 + text.len()) as u64;
        file.set_len(payload_offset + PAYLOAD_BYTES + TRAILER_BYTES)
            .unwrap();
        drop(file);

        let metadata = read_metadata(&path).unwrap();
        assert_eq!(metadata.header.text_bytes, text.len() as u32);
        let (_, prefix) = read_text_prefix(&path, prompt.len()).unwrap();
        assert_eq!(prefix, prompt);

        let mut store = Store::open(&dir, 8192, false, Options::default()).unwrap();
        assert_eq!(store.entries().len(), 1);
        assert_eq!(
            store.entries()[0].file_size,
            payload_offset + PAYLOAD_BYTES + TRAILER_BYTES
        );
        let (_, lcp) = store.find_text_lcp(&prompt, 0, 2, 2048, 5).unwrap();
        assert_eq!(lcp, prompt.len());

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn eviction_stops_after_unlink_failure() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-evict-unlink-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        store.write(rec(b"first", 512)).unwrap();
        store.write(rec(b"second", 1024)).unwrap();

        let original_permissions = fs::metadata(&dir).unwrap().permissions();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        store.budget_bytes = 1;
        store.evict(0, None);
        let disk_files = fs::read_dir(&dir).unwrap().count();
        let remaining_tokens: Vec<_> = store.entries().iter().map(|e| e.header.tokens).collect();
        fs::set_permissions(&dir, original_permissions).unwrap();
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(disk_files, 2);
        assert_eq!(remaining_tokens, vec![1024]);
    }

    #[test]
    fn compatible_empty_trailer_reuses_existing_bytes() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-reuse-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut existing = rec(b"same prompt", 512);
        existing.header.hits = 7;
        existing.trailer = b"old trailer".to_vec();
        let path = store.write(existing).unwrap();
        let sibling = store.write(rec(b"sibling", 1024)).unwrap();
        let before = fs::read(&path).unwrap();

        let mut incoming = rec(b"same prompt", 2048).header;
        incoming.quant_bits = 4;
        incoming.reason = Reason::Continued;
        incoming.ext_flags = crate::format::EXT_RESPONSES_VISIBLE;
        incoming.ctx_size = 4096;
        incoming.created_at = 99;
        incoming.last_used = 99;
        store.budget_bytes = 1;

        let got = store
            .write_payload_file(incoming, b"same prompt", &dir.join("missing-payload"), b"")
            .unwrap();
        assert_eq!(got, path);
        assert_eq!(fs::read(&path).unwrap(), before);
        assert!(sibling.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reuse_compatible_empty_trailer_returns_existing_without_staging() {
        let dir =
            std::env::temp_dir().join(format!("ds4-kv-preflight-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut existing = rec(b"same prompt", 512);
        existing.header.hits = 7;
        existing.trailer = b"old trailer".to_vec();
        let path = store.write(existing).unwrap();
        let sibling = store.write(rec(b"sibling", 1024)).unwrap();
        let before = fs::read(&path).unwrap();
        let sibling_before = fs::read(&sibling).unwrap();

        let mut incoming = rec(b"same prompt", 2048).header;
        incoming.quant_bits = 4;
        incoming.reason = Reason::Continued;
        incoming.ext_flags = crate::format::EXT_RESPONSES_VISIBLE;
        incoming.ctx_size = 4096;
        incoming.created_at = 99;
        incoming.last_used = 99;
        store.budget_bytes = 1;

        let got = store
            .reuse_compatible(incoming, b"same prompt", b"")
            .unwrap();

        assert_eq!(got, Some(path.clone()));
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(fs::read(&sibling).unwrap(), sibling_before);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reuse_compatible_incompatible_entry_returns_none_without_mutation() {
        let dir = std::env::temp_dir().join(format!(
            "ds4-kv-preflight-incompatible-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let path = store.write(rec(b"same prompt", 512)).unwrap();
        let sibling = store.write(rec(b"sibling", 1024)).unwrap();
        let before = fs::read(&path).unwrap();
        let sibling_before = fs::read(&sibling).unwrap();
        let mut incoming = rec(b"same prompt", 512).header;
        incoming.model_id = 1;
        store.budget_bytes = 1;

        let got = store
            .reuse_compatible(incoming, b"same prompt", b"")
            .unwrap();

        assert_eq!(got, None);
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(fs::read(&sibling).unwrap(), sibling_before);
        assert_eq!(store.entries().len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reuse_compatible_nonempty_trailer_matches_writer_fast_path() {
        let dir =
            std::env::temp_dir().join(format!("ds4-kv-preflight-trailer-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut existing = rec(b"same prompt", 512);
        existing.header.hits = 7;
        existing.header.created_at = 42;
        existing.header.last_used = 43;
        existing.header.ext_flags = crate::format::EXT_RESPONSES_VISIBLE;
        existing.payload = b"opaque payload".to_vec();
        existing.trailer = b"old trailer".to_vec();
        let path = store.write(existing).unwrap();
        let sibling = store.write(rec(b"sibling", 1024)).unwrap();
        let before = read_path(&path).unwrap();

        let mut incoming = rec(b"same prompt", 2048).header;
        incoming.reason = Reason::Continued;
        incoming.ext_flags = crate::format::EXT_THINKING_VISIBLE;
        incoming.ctx_size = 4096;
        incoming.created_at = 99;
        incoming.last_used = 99;
        store.budget_bytes = 1;
        let started = now_secs();

        let got = store
            .reuse_compatible(incoming, b"same prompt", b"new tool map")
            .unwrap();
        let after = read_path(&path).unwrap();

        assert_eq!(got, Some(path));
        assert_eq!(after.text, before.text);
        assert_eq!(after.payload, before.payload);
        assert_eq!(after.trailer, b"new tool map");
        assert_eq!(after.header.model_id, before.header.model_id);
        assert_eq!(after.header.quant_bits, before.header.quant_bits);
        assert_eq!(after.header.reason, before.header.reason);
        assert_eq!(after.header.tokens, before.header.tokens);
        assert_eq!(after.header.hits, before.header.hits);
        assert_eq!(after.header.ctx_size, before.header.ctx_size);
        assert_eq!(after.header.created_at, before.header.created_at);
        assert_eq!(after.header.payload_bytes, before.header.payload_bytes);
        assert_eq!(after.header.text_bytes, before.header.text_bytes);
        assert_eq!(
            after.header.ext_flags,
            before.header.ext_flags | crate::format::EXT_TOOL_MAP
        );
        assert!(after.header.last_used >= started);
        assert!(sibling.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn compatible_trailer_rewrite_preserves_payload_and_header() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-reuse-trailer-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut existing = rec(b"same prompt", 512);
        existing.header.hits = 7;
        existing.header.created_at = 42;
        existing.header.last_used = 43;
        existing.header.ext_flags = crate::format::EXT_RESPONSES_VISIBLE;
        existing.payload = b"opaque payload".to_vec();
        existing.trailer = b"old trailer".to_vec();
        let path = store.write(existing).unwrap();
        let sibling = store.write(rec(b"sibling", 1024)).unwrap();
        let before = read_path(&path).unwrap();

        let mut incoming = rec(b"same prompt", 2048).header;
        incoming.reason = Reason::Continued;
        incoming.ext_flags = crate::format::EXT_THINKING_VISIBLE;
        incoming.ctx_size = 4096;
        incoming.created_at = 99;
        incoming.last_used = 99;
        store.budget_bytes = 1;
        let started = now_secs();

        store
            .write_payload_file(
                incoming,
                b"same prompt",
                &dir.join("missing-payload"),
                b"new tool map",
            )
            .unwrap();
        let got = read_path(&path).unwrap();
        assert_eq!(got.text, before.text);
        assert_eq!(got.payload, before.payload);
        assert_eq!(got.trailer, b"new tool map");
        assert_eq!(got.header.model_id, before.header.model_id);
        assert_eq!(got.header.quant_bits, before.header.quant_bits);
        assert_eq!(got.header.reason, before.header.reason);
        assert_eq!(got.header.tokens, before.header.tokens);
        assert_eq!(got.header.hits, before.header.hits);
        assert_eq!(got.header.ctx_size, before.header.ctx_size);
        assert_eq!(got.header.created_at, before.header.created_at);
        assert_eq!(got.header.payload_bytes, before.header.payload_bytes);
        assert_eq!(got.header.text_bytes, before.header.text_bytes);
        assert_eq!(
            got.header.ext_flags,
            before.header.ext_flags | crate::format::EXT_TOOL_MAP
        );
        assert!(got.header.last_used >= started);
        assert!(sibling.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn payload_destination_alias_is_rejected_without_data_loss() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-alias-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let path = store.write(rec(b"same prompt", 512)).unwrap();
        let before = fs::read(&path).unwrap();
        let mut incompatible = rec(b"same prompt", 512).header;
        incompatible.model_id = 1;

        let error = store
            .write_payload_file(incompatible, b"same prompt", &path, b"")
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(fs::read(&path).unwrap(), before);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn stream_truncation_maps_to_unexpected_eof() {
        assert_eq!(
            format_io_error(FormatError::Truncated).kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn failed_incompatible_stage_preserves_existing_entries() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-stage-fail-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let target = store.write(rec(b"same prompt", 8192)).unwrap();
        let sibling = store.write(rec(b"sibling", 512)).unwrap();
        let target_before = fs::read(&target).unwrap();
        let sibling_before = fs::read(&sibling).unwrap();
        let mut incoming = rec(b"same prompt", 1024).header;
        incoming.model_id = 1;

        let error = store
            .write_payload_file(incoming, b"same prompt", &dir.join("missing-payload"), b"")
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert_eq!(fs::read(&target).unwrap(), target_before);
        assert_eq!(fs::read(&sibling).unwrap(), sibling_before);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn incompatible_replacement_credits_target_and_keeps_sibling() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-replace-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let text = b"same prompt";
        let target = store.write(rec(text, 8192)).unwrap();
        let sibling = store.write(rec(b"sibling", 512)).unwrap();
        let payload = b"new payload";
        let payload_path = dir.join("incoming.payload");
        fs::write(&payload_path, payload).unwrap();
        let incoming_bytes = file_size_bytes(text.len() as u64, payload.len() as u64, 0).unwrap();
        store.budget_bytes = fs::metadata(&sibling).unwrap().len() + incoming_bytes;
        let mut incoming = rec(text, 1024).header;
        incoming.model_id = 1;

        let got = store
            .write_payload_file(incoming, text, &payload_path, b"")
            .unwrap();
        let replaced = read_path(&target).unwrap();
        assert_eq!(got, target);
        assert_eq!(replaced.header.model_id, 1);
        assert_eq!(replaced.payload, payload);
        assert!(sibling.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_reopens_and_validates_text() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-candidate-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let path = store.write(rec(b"shared prefix", 512)).unwrap();

        let (got_path, envelope) = store
            .text_prefix_candidate(b"shared prefix and suffix", 0, 2, 8192)
            .unwrap()
            .unwrap();
        assert_eq!(got_path, path);
        assert_eq!(envelope.text, b"shared prefix");

        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start((crate::format::FIXED_HEADER + 4) as u64))
            .unwrap();
        file.write_all(b"X").unwrap();
        file.flush().unwrap();
        // The text no longer hashes to its own name: the record is refused
        // by its contents, which is not the same as not being there.
        assert_eq!(
            store
                .text_prefix_candidate(b"shared prefix and suffix", 0, 2, 8192)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn bank_candidate_requires_a_valid_strict_suffix() {
        let dir =
            std::env::temp_dir().join(format!("ds4-kv-bank-candidate-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut record = rec(b"shared prefix", 512);
        record.header.reason = Reason::BankShutdown;
        record.header.ext_flags = crate::format::EXT_BANK_REPLAY_V1;
        let path = store.write(record).unwrap();
        let mut serial = rec(b"serial exact", 512);
        serial.header.reason = Reason::Shutdown;
        let serial_path = store.write(serial).unwrap();

        assert!(store
            .bank_text_prefix_candidate(b"shared prefix", 0, 2, 8192)
            .unwrap()
            .is_none());
        assert!(store
            .bank_text_prefix_candidate(b"serial exact", 0, 2, 8192)
            .unwrap()
            .is_none());
        assert_eq!(
            store
                .bank_text_prefix_candidate(b"serial exact suffix", 0, 2, 8192)
                .unwrap()
                .unwrap()
                .0,
            serial_path
        );
        let (got_path, envelope) = store
            .bank_text_prefix_candidate(b"shared prefix and suffix", 0, 2, 8192)
            .unwrap()
            .unwrap();
        assert_eq!(got_path, path);
        assert_eq!(envelope.text, b"shared prefix");

        let mut file = fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start((crate::format::FIXED_HEADER + 4) as u64))
            .unwrap();
        file.write_all(b"X").unwrap();
        file.flush().unwrap();
        assert_eq!(
            store
                .bank_text_prefix_candidate(b"shared prefix and suffix", 0, 2, 8192)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn touch_hit_preserves_payload_and_trailer() {
        let dir = std::env::temp_dir().join(format!("ds4-kv-touch-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut store = Store::open(&dir, 16, false, Options::default()).unwrap();
        let mut record = rec(b"shared prefix", 512);
        record.header.hits = 7;
        record.header.created_at = 42;
        record.header.last_used = 43;
        record.payload = b"opaque payload".to_vec();
        record.trailer = b"trailer".to_vec();
        let path = store.write(record).unwrap();
        let before = read_path(&path).unwrap();
        let started = now_secs();

        store.touch_hit(&path).unwrap();
        let after = read_path(&path).unwrap();
        assert_eq!(after.header.hits, before.header.hits + 1);
        assert_eq!(after.header.created_at, before.header.created_at);
        assert!(after.header.last_used >= started);
        assert_eq!(after.text, before.text);
        assert_eq!(after.payload, before.payload);
        assert_eq!(after.trailer, before.trailer);
        let _ = fs::remove_dir_all(&dir);
    }
}
