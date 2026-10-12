//! The V4.1 disk half (item 4): the checkpoint payload in the KVC disk
//! store.
//!
//! The store is crates/ds4-kv (the port of ds4_kvstore.c): file format, SHA
//! key, budget, eviction and the artifact identity are its contract, never
//! duplicated here.  This module is the V4.1 caller's half:
//!
//! - the store's text block is the token array (LE i32), so its byte-prefix
//!   match is an exact token-prefix match; the native restore re-runs its
//!   own memcmp (ds4_ds41_kvreuse.inc), so this match only selects;
//! - the trailer carries what the store cannot know: the payload ABI, the
//!   packed-KV class, and the mount digest (the amplifier identity the
//!   native drops the in-memory checkpoint on, v41_run.rs);
//! - `store` stages the native payload export (a temp file the native
//!   writes) through write_payload_file, which owns the atomic rename and
//!   the eviction; `find` answers the payload range the native imports.

use std::path::Path;

use ds4_core::{V41KvDisk, V41KvHit};
use ds4_kv::{read_trailer, Store as KvStore};

/// The quant slot the store header carries.  The V4.1 payload is the packed
/// FP4-nibble + E4M3-scale KV (ds41_kvfmt.h), so the class is 4 -- one of
/// the quant classes ds4_kvstore_quant_bits_supported admits.  A class
/// marker, not an artifact quant.
const KV_QUANT_BITS: u8 = 4;

/// The V4.1 trailer (12 bytes; the store appends its identity block after
/// it): [0] payload ABI (the span order in ds4_ds41_kvreuse.inc; bump on
/// change), [1] packed-KV class (1 = ds41_kvfmt.h 288/72), [4..12] the
/// mount digest LE.
const TRAILER_BYTES: usize = 12;
const TRAILER_ABI: u8 = 1;
const TRAILER_CLASS: u8 = 1;

/// The trailer read cap: ours plus the store's identity block and slack.
const TRAILER_READ_MAX: u64 = 4096;

pub struct V41Disk<'a> {
    store: &'a mut KvStore,
    model_id: u8,
    ctx: u32,
}

impl<'a> V41Disk<'a> {
    pub fn new(store: &'a mut KvStore, model_id: u8, ctx: u32) -> Self {
        Self {
            store,
            model_id,
            ctx,
        }
    }
}

/// The store's text block for a V4.1 checkpoint: the token array as LE i32.
/// A byte prefix at 4-byte granularity is exactly a token prefix.
fn tokens_text(tokens: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        out.extend_from_slice(&token.to_le_bytes());
    }
    out
}

fn trailer(digest: u64) -> [u8; TRAILER_BYTES] {
    let mut out = [0u8; TRAILER_BYTES];
    out[0] = TRAILER_ABI;
    out[1] = TRAILER_CLASS;
    out[4..12].copy_from_slice(&digest.to_le_bytes());
    out
}

/// The V4.1 trailer check: ABI, class, mount digest.  The store's own
/// header/identity checks (model, quant class, ctx, artifact identity) have
/// already run in the candidate lookup; this is the mount identity the store
/// cannot know.
fn trailer_ok(raw: &[u8], digest: u64) -> bool {
    raw.len() >= TRAILER_BYTES
        && raw[0] == TRAILER_ABI
        && raw[1] == TRAILER_CLASS
        && u64::from_le_bytes(raw[4..12].try_into().unwrap()) == digest
}

impl V41KvDisk for V41Disk<'_> {
    fn find(&mut self, tokens: &[i32], digest: u64, have: u32) -> Option<V41KvHit> {
        let text = tokens_text(tokens);
        let (path, env) = self
            .store
            .text_prefix_candidate(&text, self.model_id, KV_QUANT_BITS, self.ctx)
            .ok()??;
        // The store's token count must not exceed this prompt: the native
        // import copies that many tokens from the caller's slice.
        if env.header.tokens <= have || env.header.tokens as usize > tokens.len() {
            return None;
        }
        let (_, raw) = read_trailer(&path, TRAILER_READ_MAX).ok()?;
        if !trailer_ok(&raw, digest) {
            return None;
        }
        Some(V41KvHit {
            path,
            offset: env.payload_offset,
            bytes: env.header.payload_bytes,
            tokens: env.header.tokens,
        })
    }

    fn store(&mut self, tokens: &[i32], digest: u64, save: &mut dyn FnMut(&Path) -> bool) {
        if (tokens.len() as i64) < i64::from(self.store.opt.min_tokens) {
            return;
        }
        let text = tokens_text(tokens);
        let trailer = trailer(digest);
        // An entry under this exact text is current only when ITS trailer
        // carries the same mount digest.  A different mount leaves a stale
        // payload behind, and the store's reuse fast path would rewrite only
        // the trailer (keeping the old mount's KV), so a stale entry is
        // discarded before the write.  A shorter prefix entry is left alone:
        // it is a different checkpoint (a different name).
        if let Ok(Some((path, env))) =
            self.store
                .text_prefix_candidate(&text, self.model_id, KV_QUANT_BITS, self.ctx)
        {
            if env.text == text {
                let current = read_trailer(&path, TRAILER_READ_MAX)
                    .map(|(_, raw)| trailer_ok(&raw, digest))
                    .unwrap_or(false);
                if current {
                    return;
                }
                let _ = self.store.discard(&path);
            }
        }
        let Ok(temp) = self.store.payload_temp() else {
            return;
        };
        if !save(temp.path()) {
            return;
        }
        // kv_header is the port's header builder (generate.rs; the session
        // route stores with the same shape).  payload/text bytes are filled
        // by write_payload_file.
        let header =
            crate::generate::kv_header(self.model_id, KV_QUANT_BITS, self.ctx, tokens.len() as u32);
        let _ = self
            .store
            .write_payload_file(header, &text, temp.path(), &trailer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_kv::Options;
    use std::path::PathBuf;

    fn tmpdir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ds4-v41-disk-{n}-{name}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn open_store(dir: &Path, min_tokens: i32) -> KvStore {
        let opt = Options {
            min_tokens,
            ..Options::default()
        };
        KvStore::open(dir, 16, false, opt).unwrap()
    }

    fn save_zeros(path: &Path, n: usize) -> bool {
        std::fs::write(path, vec![0u8; n]).is_ok()
    }

    fn entries(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .and_then(ds4_kv::sha_hex_name)
                    .is_some()
            })
            .collect()
    }

    #[test]
    fn tokens_text_is_le_i32() {
        assert_eq!(
            tokens_text(&[1, -2]),
            vec![1, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff]
        );
    }

    #[test]
    fn store_then_find_roundtrip() {
        let dir = tmpdir("roundtrip");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, 0xabc, &mut save);

        let hit = disk.find(&tokens, 0xabc, 0).expect("hit");
        assert_eq!(hit.tokens, 40);
        assert_eq!(hit.bytes, 128);
        assert!(hit.path.exists());

        // The in-memory checkpoint already at this length leaves nothing to
        // gain; a different mount digest refuses.
        assert!(disk.find(&tokens, 0xabc, 40).is_none());
        assert!(disk.find(&tokens, 0xdef, 0).is_none());

        // A longer prompt matches the 40-token prefix; an unrelated one is a
        // miss.
        let mut longer = tokens.clone();
        longer.push(999);
        assert_eq!(disk.find(&longer, 0xabc, 0).expect("prefix").tokens, 40);
        let other: Vec<i32> = (500..540).collect();
        assert!(disk.find(&other, 0xabc, 0).is_none());
    }

    #[test]
    fn current_entry_is_not_rewritten() {
        let dir = tmpdir("current");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save_small = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, 0xabc, &mut save_small);
        // A second store under the same digest must skip the export (a
        // rewrite would carry the new size).
        let mut save_big = |p: &Path| save_zeros(p, 256);
        disk.store(&tokens, 0xabc, &mut save_big);
        assert_eq!(entries(&dir).len(), 1);
        assert_eq!(disk.find(&tokens, 0xabc, 0).expect("hit").bytes, 128);
    }

    #[test]
    fn stale_mount_entry_is_replaced() {
        let dir = tmpdir("stale");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save_small = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, 0xabc, &mut save_small);
        // The mount changed: the old payload is the old amplifier's KV and
        // must not survive under the new trailer.
        let mut save_big = |p: &Path| save_zeros(p, 256);
        disk.store(&tokens, 0xdef, &mut save_big);
        assert_eq!(entries(&dir).len(), 1);
        assert!(disk.find(&tokens, 0xabc, 0).is_none());
        assert_eq!(disk.find(&tokens, 0xdef, 0).expect("hit").bytes, 256);
    }

    #[test]
    fn min_tokens_gate_skips_short_prompts() {
        let dir = tmpdir("min-tokens");
        let mut store = open_store(&dir, 64);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, 0xabc, &mut save);
        assert!(entries(&dir).is_empty());
        assert!(disk.find(&tokens, 0xabc, 0).is_none());
    }

    #[test]
    fn find_works_with_bound_identity() {
        let dir = tmpdir("identity");
        let mut store = open_store(&dir, 4);
        store.bind_identity([7; 32]);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, 0xabc, &mut save);
        // The identity block follows our 12-byte trailer; the check must
        // read only its own prefix.
        assert_eq!(disk.find(&tokens, 0xabc, 0).expect("hit").tokens, 40);
        // A store bound to a different identity refuses the entry.
        let mut other = open_store(&dir, 4);
        other.bind_identity([9; 32]);
        let mut other_disk = V41Disk::new(&mut other, 7, 4096);
        assert!(other_disk.find(&tokens, 0xabc, 0).is_none());
    }
}
