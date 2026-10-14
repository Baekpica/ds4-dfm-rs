//! The V4.1 disk half (item 4): the checkpoint payload in the KVC disk
//! store.
//!
//! The store is crates/ds4-kv (the port of ds4_kvstore.c): file format, SHA
//! key, budget, eviction and the artifact identity are its contract, never
//! duplicated here.  This module is the V4.1 caller's half:
//!
//! - the store's text block interleaves each token (LE i32) with its image
//!   signature (LE u64, zero for text), so its byte-prefix match preserves
//!   image identity and order; native restore repeats the comparison
//!   (ds4_ds41_kvreuse.inc), so this match only selects;
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
const TOKEN_KEY_BYTES: usize = std::mem::size_of::<i32>() + std::mem::size_of::<u64>();

/// The V4.1 trailer (12 bytes; the store appends its identity block after
/// it): [0] payload ABI (the span order in ds4_ds41_kvreuse.inc; bump on
/// change), [1] packed-KV class (1 = ds41_kvfmt.h 288/72), [4..12] the
/// mount digest LE.
const TRAILER_BYTES: usize = 12;
const TRAILER_ABI: u8 = 2;
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

/// Interleave each token with its image signature (zero for text). Prefix
/// lookup then rejects changed images while admitting later appended images.
fn tokens_text(tokens: &[i32], sig: &[u64]) -> Option<Vec<u8>> {
    // An empty slice denotes text. A partial image map must never alias it.
    if !sig.is_empty() && sig.len() != tokens.len() {
        return None;
    }

    let mut out = Vec::with_capacity(tokens.len().checked_mul(TOKEN_KEY_BYTES)?);
    for (i, token) in tokens.iter().enumerate() {
        let image_sig = if sig.is_empty() { 0 } else { sig[i] };
        out.extend_from_slice(&token.to_le_bytes());
        out.extend_from_slice(&image_sig.to_le_bytes());
    }
    Some(out)
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
    fn find(&mut self, tokens: &[i32], sig: &[u64], digest: u64, have: u32) -> Option<V41KvHit> {
        let text = tokens_text(tokens, sig)?;
        let (path, env) = self
            .store
            .text_prefix_candidate(&text, self.model_id, KV_QUANT_BITS, self.ctx)
            .ok()??;
        // The store's token count must not exceed this prompt: the native
        // import copies that many tokens from the caller's slice.
        if env.header.tokens <= have
            || env.header.tokens as usize > tokens.len()
            || env.text.len() != env.header.tokens as usize * TOKEN_KEY_BYTES
        {
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

    fn store(
        &mut self,
        tokens: &[i32],
        sig: &[u64],
        digest: u64,
        save: &mut dyn FnMut(&Path) -> bool,
    ) {
        if (tokens.len() as i64) < i64::from(self.store.opt.min_tokens) {
            return;
        }
        let Some(text) = tokens_text(tokens, sig) else {
            return;
        };
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
    fn key_interleaves_token_and_sig() {
        assert_eq!(
            tokens_text(&[1, -2], &[0x0807_0605_0403_0201, 0]).unwrap(),
            vec![
                1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0xfe, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(tokens_text(&[1, -2], &[]), tokens_text(&[1, -2], &[0; 2]));
    }

    #[test]
    fn store_then_find_roundtrip() {
        let dir = tmpdir("roundtrip");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, &[], 0xabc, &mut save);

        let hit = disk.find(&tokens, &[], 0xabc, 0).expect("hit");
        assert_eq!(hit.tokens, 40);
        assert_eq!(hit.bytes, 128);
        assert!(hit.path.exists());

        // The in-memory checkpoint already at this length leaves nothing to
        // gain; a different mount digest refuses.
        assert!(disk.find(&tokens, &[], 0xabc, 40).is_none());
        assert!(disk.find(&tokens, &[], 0xdef, 0).is_none());

        // A longer prompt matches the 40-token prefix; an unrelated one is a
        // miss.
        let mut longer = tokens.clone();
        longer.push(999);
        assert_eq!(
            disk.find(&longer, &[], 0xabc, 0).expect("prefix").tokens,
            40
        );
        let other: Vec<i32> = (500..540).collect();
        assert!(disk.find(&other, &[], 0xabc, 0).is_none());
    }

    #[test]
    fn image_prefix_survives_append() {
        let dir = tmpdir("append-image");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens = [100, 900, 900, 101, 900, 900, 102];
        let sig = [0, 11, 12, 0, 21, 22, 0];
        disk.store(&tokens, &sig, 0xabc, &mut |p| save_zeros(p, 128));

        let mut longer = tokens.to_vec();
        longer.extend_from_slice(&[900, 900, 103]);
        let mut longer_sig = sig.to_vec();
        longer_sig.extend_from_slice(&[31, 32, 0]);
        let hit = disk
            .find(&longer, &longer_sig, 0xabc, 0)
            .expect("an appended image preserves the stored prefix");
        assert_eq!(hit.tokens, tokens.len() as u32);
        assert_eq!(hit.bytes, 128);
    }

    #[test]
    fn changed_images_miss() {
        let dir = tmpdir("changed-images");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens = [100, 900, 900, 101, 900, 900, 102];
        let sig = [0, 11, 12, 0, 21, 22, 0];
        disk.store(&tokens, &sig, 0xabc, &mut |p| save_zeros(p, 128));

        let replacement = [0, 31, 32, 0, 21, 22, 0];
        let reordered = [0, 21, 22, 0, 11, 12, 0];
        assert!(disk.find(&tokens, &replacement, 0xabc, 0).is_none());
        assert!(disk.find(&tokens, &reordered, 0xabc, 0).is_none());
        assert!(disk.find(&tokens, &[], 0xabc, 0).is_none());
        assert!(disk.find(&tokens, &sig, 0xabc, 0).is_some());
    }

    #[test]
    fn image_identity_survives_reopen() {
        let dir = tmpdir("reopen-image");
        let tokens = [100, 900, 900, 101];
        let sig = [0, 11, 12, 0];
        let original = {
            let mut store = open_store(&dir, 4);
            store.bind_identity([7; 32]);
            let mut disk = V41Disk::new(&mut store, 7, 4096);
            disk.store(&tokens, &sig, 0xabc, &mut |p| save_zeros(p, 128));
            disk.find(&tokens, &sig, 0xabc, 0).expect("stored").path
        };

        let mut reopened = open_store(&dir, 4);
        reopened.bind_identity([7; 32]);
        let mut disk = V41Disk::new(&mut reopened, 7, 4096);
        let hit = disk.find(&tokens, &sig, 0xabc, 0).expect("reopened");
        assert_eq!(hit.path, original);
        assert_eq!(hit.bytes, 128);
        assert!(disk.find(&tokens, &[0, 21, 22, 0], 0xabc, 0).is_none());
        assert!(disk.find(&tokens, &sig, 0xdef, 0).is_none());
        drop(reopened);

        let mut other = open_store(&dir, 4);
        other.bind_identity([9; 32]);
        assert!(V41Disk::new(&mut other, 7, 4096)
            .find(&tokens, &sig, 0xabc, 0)
            .is_none());
    }

    #[test]
    fn legacy_payload_abi_is_rejected() {
        let dir = tmpdir("legacy-abi");
        let mut store = open_store(&dir, 4);
        let tokens = [100, 101, 102, 103];
        let mut old_trailer = trailer(0xabc);
        old_trailer[0] = 1;
        let payload = store.payload_temp().unwrap();
        assert!(save_zeros(payload.path(), 128));
        let header = crate::generate::kv_header(7, KV_QUANT_BITS, 4096, tokens.len() as u32);
        // Use the current key to prove ABI rejection independently of the
        // changed key width: even an otherwise matching legacy payload misses.
        store
            .write_payload_file(
                header,
                &tokens_text(&tokens, &[]).unwrap(),
                payload.path(),
                &old_trailer,
            )
            .unwrap();
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        assert!(disk.find(&tokens, &[], 0xabc, 0).is_none());

        disk.store(&tokens, &[], 0xabc, &mut |p| save_zeros(p, 256));
        assert_eq!(
            disk.find(&tokens, &[], 0xabc, 0).expect("new ABI").bytes,
            256
        );
        assert_eq!(entries(&dir).len(), 1);
    }

    #[test]
    fn malformed_signatures_miss() {
        let dir = tmpdir("malformed-find");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens = [100, 101, 102, 103];
        disk.store(&tokens, &[], 0xabc, &mut |p| save_zeros(p, 128));
        let invalid: [&[u64]; 2] = [&[0; 3], &[0; 5]];
        let hits: Vec<_> = invalid
            .into_iter()
            .filter(|sig| disk.find(&tokens, sig, 0xabc, 0).is_some())
            .map(<[u64]>::len)
            .collect();
        assert!(hits.is_empty(), "invalid signature lengths hit: {hits:?}");
        assert!(disk.find(&tokens, &[], 0xabc, 0).is_some());
        assert!(disk.find(&tokens, &[0; 4], 0xabc, 0).is_some());
    }

    #[test]
    fn malformed_signatures_not_saved() {
        let dir = tmpdir("malformed-store");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens = [100, 101, 102, 103];
        let invalid: [&[u64]; 2] = [&[0; 3], &[0; 5]];
        let mut exports = 0;
        for sig in invalid {
            disk.store(&tokens, sig, sig.len() as u64, &mut |p| {
                exports += 1;
                save_zeros(p, 128)
            });
        }
        assert_eq!(exports, 0, "invalid signatures reached native export");
        assert!(entries(&dir).is_empty());
    }

    #[test]
    fn current_entry_is_not_rewritten() {
        let dir = tmpdir("current");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save_small = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, &[], 0xabc, &mut save_small);
        // A second store under the same digest must skip the export (a
        // rewrite would carry the new size).
        let mut save_big = |p: &Path| save_zeros(p, 256);
        disk.store(&tokens, &[], 0xabc, &mut save_big);
        assert_eq!(entries(&dir).len(), 1);
        assert_eq!(disk.find(&tokens, &[], 0xabc, 0).expect("hit").bytes, 128);
    }

    #[test]
    fn stale_mount_entry_is_replaced() {
        let dir = tmpdir("stale");
        let mut store = open_store(&dir, 4);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save_small = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, &[], 0xabc, &mut save_small);
        // The mount changed: the old payload is the old amplifier's KV and
        // must not survive under the new trailer.
        let mut save_big = |p: &Path| save_zeros(p, 256);
        disk.store(&tokens, &[], 0xdef, &mut save_big);
        assert_eq!(entries(&dir).len(), 1);
        assert!(disk.find(&tokens, &[], 0xabc, 0).is_none());
        assert_eq!(disk.find(&tokens, &[], 0xdef, 0).expect("hit").bytes, 256);
    }

    #[test]
    fn min_tokens_gate_skips_short_prompts() {
        let dir = tmpdir("min-tokens");
        let mut store = open_store(&dir, 64);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, &[], 0xabc, &mut save);
        assert!(entries(&dir).is_empty());
        assert!(disk.find(&tokens, &[], 0xabc, 0).is_none());
    }

    #[test]
    fn find_works_with_bound_identity() {
        let dir = tmpdir("identity");
        let mut store = open_store(&dir, 4);
        store.bind_identity([7; 32]);
        let mut disk = V41Disk::new(&mut store, 7, 4096);
        let tokens: Vec<i32> = (100..140).collect();
        let mut save = |p: &Path| save_zeros(p, 128);
        disk.store(&tokens, &[], 0xabc, &mut save);
        // The identity block follows our 12-byte trailer; the check must
        // read only its own prefix.
        assert_eq!(disk.find(&tokens, &[], 0xabc, 0).expect("hit").tokens, 40);
        // A store bound to a different identity refuses the entry.
        let mut other = open_store(&dir, 4);
        other.bind_identity([9; 32]);
        let mut other_disk = V41Disk::new(&mut other, 7, 4096);
        assert!(other_disk.find(&tokens, &[], 0xabc, 0).is_none());
    }
}
