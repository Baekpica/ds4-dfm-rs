//! Live bank-evict persist: write `KvReason::BankEvict` before destroy.
//!
//! C `kv_cache_store_bank(..., "bank-evict", warm_pin_min)` then
//! `warm_rec_invalidate`. A continuation-protected (pinned) bank is never
//! a victim.
//!
//! The save path is the same `save_bank_record` ContLane::persist_bank uses;
//! only the reason is `BankEvict`. Live place_* call ContLane::evict_bank,
//! which is persist_bank(BankEvict) plus this pin-skip / drop contract.

use std::path::Path;

use ds4_kv::{Reason as KvReason, Store as KvStore};

use crate::generate::GenerateError;
#[cfg(test)]
use crate::serve_cont::WarmRecord;
use crate::serve_cont::{save_bank_record, WarmBank};

/// Persist a pin-tier bank with `BankEvict`, then drop its warm record.
///
/// Returns `Ok(false)` and leaves the bank untouched when `pinned`.
pub(crate) fn persist_bank_evict(
    store: &mut KvStore,
    warm: &mut WarmBank,
    pinned: bool,
    identity: (u8, u8, u32),
    committed: i32,
    generation: u64,
    pin_min: i32,
    save_payload: impl FnOnce(&Path) -> Result<(), GenerateError>,
) -> Result<bool, GenerateError> {
    if pinned {
        return Ok(false);
    }
    if pin_min > 0 && committed >= pin_min {
        if let Err(error) = save_bank_record(
            store,
            warm,
            identity,
            committed,
            generation,
            KvReason::BankEvict,
            save_payload,
        ) {
            eprintln!("ds4-server-rs: bank checkpoint failed reason=bank-evict: {error}");
        }
    }
    warm.record = None;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4_kv::{Options, Reason, EXT_BANK_REPLAY_V1, EXT_RESPONSES_VISIBLE, EXT_SESSION_TITLE};
    use std::fs;

    fn temp_store(tag: &str) -> (std::path::PathBuf, KvStore) {
        let dir = std::env::temp_dir().join(format!(
            "ds4-server-bank-evict-{tag}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let store = KvStore::open(&dir, 16, false, Options::default()).unwrap();
        (dir, store)
    }

    fn deep_warm(text: &str) -> WarmBank {
        WarmBank {
            record: Some(WarmRecord {
                text: text.as_bytes().to_vec(),
                cache_text: None,
                exact_text: None,
                exact_cache_text: None,
                partial_only: false,
                generation: 3,
                ext_flags: 0,
                trailer: Vec::new(),
            }),
            committed_tokens: 70_000,
            stored_tokens: 0,
            last_use: 1,
        }
    }

    #[test]
    fn evict_writes_file_with_bank_evict_reason() {
        // Given: an unpinned pin-tier bank with a matching generation
        let (dir, mut store) = temp_store("write");
        let mut warm = deep_warm("evict-prefix");

        // When: live evict persist runs at the pin threshold
        let wrote = persist_bank_evict(
            &mut store,
            &mut warm,
            false,
            (1, 2, 8192),
            70_000,
            3,
            65_536,
            |path| {
                fs::write(path, b"bank-payload").map_err(|e| GenerateError::Engine(e.to_string()))
            },
        )
        .unwrap();

        // Then: a KVC file is stored with reason BankEvict and the record is gone
        assert!(wrote);
        assert!(warm.record.is_none());
        assert_eq!(store.entries().len(), 1);
        let record = store.read(&store.entries()[0].path).unwrap();
        assert_eq!(record.header.reason, Reason::BankEvict);
        assert_eq!(record.header.ext_flags, EXT_BANK_REPLAY_V1);
        assert_eq!(record.header.tokens, 70_000);
        assert_eq!(record.text, b"evict-prefix");
        assert_eq!(record.payload, b"bank-payload");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn evict_skips_a_pinned_bank() {
        // Given: a continuation-protected (pinned) pin-tier bank
        let (dir, mut store) = temp_store("pinned");
        let mut warm = deep_warm("pinned-prefix");

        // When: evict persist is asked to destroy that bank
        let wrote = persist_bank_evict(
            &mut store,
            &mut warm,
            true,
            (1, 2, 8192),
            70_000,
            3,
            65_536,
            |_| panic!("pinned banks must not stage a payload"),
        )
        .unwrap();

        // Then: no store write and the warm record stays
        assert!(!wrote);
        assert!(store.entries().is_empty());
        assert_eq!(
            warm.record.as_ref().map(|record| record.text.as_slice()),
            Some(b"pinned-prefix".as_slice())
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn evict_persist_keeps_responses_visible_and_session_title() {
        // Given: an unpinned bank whose live ext bits include Responses/title
        let (dir, mut store) = temp_store("ext");
        let mut warm = deep_warm("ext-prefix");
        if let Some(record) = warm.record.as_mut() {
            record.ext_flags = EXT_RESPONSES_VISIBLE | EXT_SESSION_TITLE;
        }

        // When: evict persist writes the pin-tier snapshot
        persist_bank_evict(
            &mut store,
            &mut warm,
            false,
            (1, 2, 8192),
            70_000,
            3,
            65_536,
            |path| fs::write(path, b"p").map_err(|e| GenerateError::Engine(e.to_string())),
        )
        .unwrap();

        // Then: BankEvict is stored and those two bits are not clobbered
        let record = store.read(&store.entries()[0].path).unwrap();
        assert_eq!(record.header.reason, Reason::BankEvict);
        assert_eq!(
            record.header.ext_flags,
            EXT_BANK_REPLAY_V1 | EXT_RESPONSES_VISIBLE | EXT_SESSION_TITLE
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn evict_does_not_write_below_pin_min() {
        // Given: an unpinned shallow bank (below warm_pin_min)
        let (dir, mut store) = temp_store("shallow");
        let mut warm = deep_warm("shallow-prefix");
        warm.committed_tokens = 10;

        // When: evict persist runs with C's pin-tier min_committed
        let wrote = persist_bank_evict(
            &mut store,
            &mut warm,
            false,
            (1, 2, 8192),
            10,
            3,
            65_536,
            |_| panic!("shallow banks are not the pin-tier persist set"),
        )
        .unwrap();

        // Then: the record is still evicted, but no BankEvict file is written
        assert!(wrote);
        assert!(warm.record.is_none());
        assert!(store.entries().is_empty());
        let _ = KvReason::BankEvict;
        let _ = fs::remove_dir_all(dir);
    }
}
