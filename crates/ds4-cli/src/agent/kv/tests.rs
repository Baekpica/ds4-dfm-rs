use super::{
    cache_dir_from_home, identity_sha, parse_slash, ListedSession, SaveSpec, SessionStore, SlashCmd,
};
use ds4_kv::decode_file;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn create() -> Self {
        let root = std::env::temp_dir().join(format!(
            "ds4_agent_kv_{}_{}",
            std::process::id(),
            TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    fn store(&self, model_id: u8) -> SessionStore {
        SessionStore::open(&self.root, model_id, [1; 32])
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn save_named(
    store: &SessionStore,
    title: &str,
    created_at: u64,
    last_used: u64,
    payload: &[u8],
) -> String {
    let temp = ds4_kv::PayloadTemp::create(store.dir()).unwrap();
    std::fs::write(temp.path(), payload).unwrap();
    store
        .save(SaveSpec {
            title,
            created_at,
            last_used,
            text: format!("<｜User｜>{title}<｜Assistant｜>ok").as_bytes(),
            payload: temp.path(),
            tokens: 8,
            model_id: 1,
            quant_bits: 2,
            ctx_size: 1024,
        })
        .unwrap()
}

#[test]
fn default_cache_dir_is_home_ds4_kvcache() {
    // Given: HOME like C getenv, or empty fallback "."
    // When: resolve the agent cache dir
    // Then: `$HOME/.ds4/kvcache` — not a new directory
    assert_eq!(
        cache_dir_from_home(Some("/home/alice".into())),
        PathBuf::from("/home/alice/.ds4/kvcache")
    );
    assert_eq!(
        cache_dir_from_home(Some("".into())),
        PathBuf::from("./.ds4/kvcache")
    );
    assert_eq!(cache_dir_from_home(None), PathBuf::from("./.ds4/kvcache"));
}

#[test]
fn list_orders_by_recent_then_sha() {
    // Given: three same-model sessions; older last_used first on disk
    let fx = Fixture::create();
    let store = fx.store(1);
    let older = save_named(&store, "older", 100, 100, b"PAY1");
    let newer = save_named(&store, "newer", 200, 300, b"PAY2");
    let tie_b = save_named(&store, "tie-b", 150, 200, b"PAY3");
    let tie_a = save_named(&store, "tie-a", 160, 200, b"PAY4");
    let temp = ds4_kv::PayloadTemp::create(store.dir()).unwrap();
    std::fs::write(temp.path(), b"P").unwrap();
    let _other_model = store
        .save(SaveSpec {
            title: "other-model",
            created_at: 400,
            last_used: 400,
            text: b"x",
            payload: temp.path(),
            tokens: 1,
            model_id: 9,
            quant_bits: 2,
            ctx_size: 1024,
        })
        .unwrap();
    std::fs::write(fx.root.join("sysprompt.kv"), b"not-a-session").unwrap();

    // When: list
    let listed = store.list().unwrap();

    // Then: last_used desc, sha asc on ties; sysprompt and other model_id dropped
    let mut tied = [tie_a.as_str(), tie_b.as_str()];
    tied.sort_unstable();
    let shas: Vec<&str> = listed.iter().map(|item| item.sha.as_str()).collect();
    assert_eq!(shas, vec![newer.as_str(), tied[0], tied[1], older.as_str()]);
    assert!(listed.iter().all(|item| !item.stripped));
    assert_eq!(listed[0].sha, newer);
    let _ = ListedSession {
        sha: newer,
        title: String::new(),
        last_used: 0,
        created_at: 0,
        tokens: 0,
        file_size: 0,
        stripped: false,
    };
}

#[test]
fn strip_then_switch_sets_prefill_flag() {
    // Given: a saved session with a KV payload
    let fx = Fixture::create();
    let store = fx.store(1);
    let sha = save_named(&store, "rebuild-me", 10, 10, b"HEAVY-KV");
    assert!(!store.switch_plan(&sha[..8]).unwrap().needs_prefill);

    // When: strip, then switch
    store.strip(&sha[..8], 12, 99).unwrap();
    let plan = store.switch_plan(&sha[..8]).unwrap();

    // Then: payload gone and switch rebuilds by prefill
    assert!(plan.needs_prefill);
    assert_eq!(plan.payload_bytes, 0);
    assert_eq!(plan.tokens, 12);
    assert_eq!(plan.sha, sha);
    let listed = store.list().unwrap();
    assert_eq!(listed.len(), 1);
    assert!(listed[0].stripped);
}

#[test]
fn del_removes_file() {
    // Given: one saved session file
    let fx = Fixture::create();
    let store = fx.store(1);
    let sha = save_named(&store, "delete-me", 1, 1, b"KV");
    let path = fx.root.join(format!("{sha}.kv"));
    assert!(path.exists());

    // When: /del by prefix
    let deleted = store.delete(&sha[..8]).unwrap();

    // Then: the file is gone
    assert_eq!(deleted, sha);
    assert!(!path.exists());
    assert!(store.list().unwrap().is_empty());
}

#[test]
fn missing_sha_errors_like_c() {
    // Given: an empty cache dir
    let fx = Fixture::create();
    let store = fx.store(1);

    // When: find/switch/del/strip a missing prefix
    // Then: C `no saved session matches %.40s`
    assert_eq!(
        store.find("deadbeef").unwrap_err().to_string(),
        "no saved session matches deadbeef"
    );
    assert_eq!(
        store.switch_plan("abc").unwrap_err().to_string(),
        "no saved session matches abc"
    );
    assert_eq!(
        store.delete("fff").unwrap_err().to_string(),
        "no saved session matches fff"
    );
    assert_eq!(
        store.strip("0", 1, 0).unwrap_err().to_string(),
        "no saved session matches 0"
    );
    assert_eq!(
        store.find("").unwrap_err().to_string(),
        "invalid session SHA prefix"
    );
    assert_eq!(
        store.find("not-hex").unwrap_err().to_string(),
        "invalid session SHA prefix"
    );
}

#[test]
fn save_keeps_kvc_magic_and_stable_identity() {
    // Given: a title + created_at identity
    let fx = Fixture::create();
    let store = fx.store(1);
    let title = "stable-id";
    let created_at = 1_700_000_042u64;
    let expected = identity_sha(title, created_at);

    // When: save twice with a growing payload
    let first = save_named(&store, title, created_at, 10, b"v1");
    let second = save_named(&store, title, created_at, 20, b"v2-longer");

    // Then: same SHA filename; file still starts with C magic KVC
    assert_eq!(first, expected);
    assert_eq!(second, expected);
    let bytes = std::fs::read(fx.root.join(format!("{expected}.kv"))).unwrap();
    assert_eq!(&bytes[..3], b"KVC");
    let record = decode_file(&bytes).unwrap();
    assert_eq!(record.payload, b"v2-longer");
}

#[test]
fn parse_slash_requires_sha_args_like_c() {
    assert_eq!(parse_slash("/save"), Some(Ok(SlashCmd::Save)));
    assert_eq!(parse_slash("/list"), Some(Ok(SlashCmd::List)));
    assert_eq!(
        parse_slash("/switch"),
        Some(Err("usage: /switch <sha-prefix>"))
    );
    assert_eq!(parse_slash("/del"), Some(Err("usage: /del <sha-prefix>")));
    assert_eq!(
        parse_slash("/strip"),
        Some(Err("usage: /strip <sha-prefix>"))
    );
    assert_eq!(
        parse_slash("/switch abcd"),
        Some(Ok(SlashCmd::Switch("abcd")))
    );
    assert_eq!(parse_slash("/help"), None);
}

#[test]
fn ambiguous_prefix_errors_like_c() {
    let fx = Fixture::create();
    let store = fx.store(1);
    let a = save_named(&store, "alpha", 1, 1, b"A");
    let b = save_named(&store, "bravo", 2, 2, b"B");
    let shared: String = a
        .chars()
        .zip(b.chars())
        .take_while(|(left, right)| left == right)
        .map(|(left, _)| left)
        .collect();
    if shared.is_empty() {
        return;
    }
    let err = store.find(&shared).unwrap_err().to_string();
    assert_eq!(err, format!("session prefix {shared:.40} is ambiguous"));
}

#[test]
fn audit_foreign_kv_rebuild() {
    for identity in [None, Some([2u8; 32])] {
        let fx = Fixture::create();
        let store = fx.store(1);
        let title = "legacy or different weights";
        let sha = identity_sha(title, 10);
        let mut trailer = super::identity::encode_title_trailer(title);
        let mut flags = ds4_kv::EXT_SESSION_TITLE;
        if let Some(identity) = identity {
            // Existing KVI footer fixture; the agent used to ignore it entirely.
            flags |= 1 << 6;
            trailer.extend_from_slice(b"KVI\x01");
            trailer.extend_from_slice(&identity);
            trailer.extend_from_slice(b"KVI!");
        }
        let text = b"<\xef\xbd\x9cUser\xef\xbd\x9c>saved conversation";
        let raw = ds4_kv::fill_header(
            1,
            2,
            ds4_kv::Reason::AgentSession,
            flags,
            8,
            0,
            1024,
            10,
            10,
            2,
        );
        let header = ds4_kv::parse_header(&raw, text.len() as u32).unwrap();
        ds4_kv::write_path(
            &ds4_kv::path_for_sha(&fx.root, &sha),
            &ds4_kv::Record {
                header,
                text: text.to_vec(),
                payload: b"KV".to_vec(),
                trailer,
            },
        )
        .unwrap();
        let plan = store.switch_plan(&sha).unwrap();
        assert!(
            plan.needs_prefill,
            "unknown artifact must rebuild from text"
        );
        assert_eq!(plan.text, text);
        assert_eq!(plan.title, title);
    }
}

#[test]
#[cfg(target_os = "linux")]
fn audit_sparse_catalog() {
    const CHILD: &str = "DS4_TEST_KV_MEMORY";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "agent::kv::tests::audit_sparse_catalog",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    use std::io::{Seek, SeekFrom, Write};
    let fx = Fixture::create();
    let store = fx.store(1);
    let text = b"saved text";
    let mut shas = Vec::new();
    for created in 1..=2 {
        let sha = identity_sha("sparse", created);
        let mut file = std::fs::File::create(ds4_kv::path_for_sha(&fx.root, &sha)).unwrap();
        let size = 8 * 1024u64.pow(3);
        file.write_all(&ds4_kv::fill_header(
            1,
            2,
            ds4_kv::Reason::AgentSession,
            ds4_kv::EXT_SESSION_TITLE,
            8,
            0,
            1024,
            created,
            created,
            size,
        ))
        .unwrap();
        file.write_all(&(text.len() as u32).to_le_bytes()).unwrap();
        file.write_all(text).unwrap();
        file.seek(SeekFrom::Current(size as i64)).unwrap();
        file.write_all(&super::identity::encode_title_trailer("sparse"))
            .unwrap();
        shas.push(sha);
    }
    // Old whole-file reads cannot fit; no 8 GiB allocation reaches the host.
    let mut old = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    unsafe {
        assert_eq!(libc::getrlimit(libc::RLIMIT_AS, &mut old), 0);
        let limit = libc::rlimit {
            rlim_cur: 512 * 1024 * 1024,
            rlim_max: old.rlim_max,
        };
        assert_eq!(libc::setrlimit(libc::RLIMIT_AS, &limit), 0);
    }
    assert_eq!(store.list().unwrap().len(), 2);
    assert_eq!(store.find(&shas[0]).unwrap().0, shas[0]);
    assert_eq!(store.switch_plan(&shas[0]).unwrap().text, text);
    store.delete(&shas[1]).unwrap();
    store.strip(&shas[0], 8, 9).unwrap();
    assert_eq!(store.switch_plan(&shas[0]).unwrap().payload_bytes, 0);
    unsafe {
        assert_eq!(libc::setrlimit(libc::RLIMIT_AS, &old), 0);
    }
}

#[test]
fn audit_identity_and_held_file() {
    let fx = Fixture::create();
    let first = fx.store(1);
    let sha = save_named(&first, "stable", 10, 10, b"first weights KV");
    let plan = first.switch_plan(&sha).unwrap();
    assert!(!plan.needs_prefill);
    let other = SessionStore::open(&fx.root, 1, [2; 32]);
    assert!(other.switch_plan(&sha).unwrap().needs_prefill);
    assert_eq!(other.list().unwrap()[0].title, "stable");
    save_named(&other, "stable", 10, 20, b"other weights KV");
    assert_eq!(
        ds4_kv::read_path(&plan.path).unwrap().payload,
        b"first weights KV"
    );
    assert_eq!(ds4_kv::read_identity(&plan.path).unwrap(), Some([1; 32]));
    assert!(first.switch_plan(&sha).unwrap().needs_prefill);
    assert!(!other.switch_plan(&sha).unwrap().needs_prefill);
    let path = ds4_kv::path_for_sha(&fx.root, &sha);
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(file.metadata().unwrap().len() - 1).unwrap();
    assert!(other.switch_plan(&sha).unwrap().needs_prefill);
}
