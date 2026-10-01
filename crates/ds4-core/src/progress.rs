use std::marker::PhantomData;
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr::NonNull;

use ds4_sys::{ds4_bridge_session, ds4_bridge_session_sync_cb};

use crate::{fail, save_payload_checked, ModelFamily, Result, Session, TokenBuffer};

/// Exact native checkpoint exposed only while a durable prefill callback runs.
pub struct PrefillCheckpoint<'a> {
    raw: NonNull<ds4_bridge_session>,
    tokens: &'a [i32],
    total: usize,
    family: ModelFamily,
    ctx: i32,
    _scope: PhantomData<&'a mut ds4_bridge_session>,
    _not_send: PhantomData<*mut ()>,
}

impl PrefillCheckpoint<'_> {
    pub fn current(&self) -> usize {
        self.tokens.len()
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn tokens(&self) -> &[i32] {
        self.tokens
    }

    pub fn save_payload(&self, path: impl AsRef<Path>) -> Result<()> {
        save_payload_checked(self.raw, path.as_ref(), self.tokens, self.family, self.ctx)
    }
}

struct SyncTramp<'a, F> {
    raw: NonNull<ds4_bridge_session>,
    tokens: &'a [i32],
    family: ModelFamily,
    ctx: i32,
    progress: &'a mut F,
}

unsafe extern "C" fn progress_tramp<F>(ud: *mut c_void, current: i32, total: i32)
where
    F: for<'a> FnMut(PrefillCheckpoint<'a>),
{
    if ud.is_null() {
        return;
    }
    let t = &mut *(ud as *mut SyncTramp<'_, F>);
    if current < 0 || total != t.tokens.len() as i32 || current > total {
        return;
    }
    let checkpoint = PrefillCheckpoint {
        raw: t.raw,
        tokens: &t.tokens[..current as usize],
        total: t.tokens.len(),
        family: t.family,
        ctx: t.ctx,
        _scope: PhantomData,
        _not_send: PhantomData,
    };
    let _ = catch_unwind(AssertUnwindSafe(|| (t.progress)(checkpoint)));
}

impl Session<'_> {
    /// Syncs once and exposes only native durable prefill frontiers.
    pub fn sync_progress<F>(&mut self, tokens: &TokenBuffer, mut progress: F) -> Result<()>
    where
        F: for<'a> FnMut(PrefillCheckpoint<'a>),
    {
        let plan = self.check_sync(tokens)?;
        let mut t = SyncTramp {
            raw: self.raw,
            tokens: tokens.as_slice(),
            family: self.host.family,
            ctx: self.host.ctx,
            progress: &mut progress,
        };
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_session_sync_cb(
                self.raw.as_ptr(),
                tokens.as_slice().as_ptr(),
                tokens.len() as i32,
                Some(progress_tramp::<F>),
                &mut t as *mut SyncTramp<'_, F> as *mut c_void,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            self.step_failed();
            return Err(fail(rc, &err));
        }
        self.host.commit_sync(tokens.as_slice(), &plan);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::ffi::CStr;
    use std::marker::PhantomData;
    use std::os::raw::{c_char, c_void};
    use std::ptr::NonNull;

    use ds4_sys::ds4_bridge_session;

    use crate::{
        HostPrefix, ModelFamily, Session, SessionBackend, SessionLedger, TokenBuffer,
        PAYLOAD_MAGIC, PAYLOAD_VERSION,
    };

    thread_local! {
        static SYNC_RC: Cell<i32> = const { Cell::new(0) };
        static SYNC_FAILURE: Cell<SyncFailure> = const { Cell::new(SyncFailure::None) };
        static NATIVE_PREFIX: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum SyncFailure {
        None,
        Rejected,
        Mutated,
    }

    #[no_mangle]
    extern "C" fn ds4_bridge_session_exaone_rewind_span(s: *mut ds4_bridge_session) -> i32 {
        assert!(!s.is_null());
        0
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_session_sync_cb(
        s: *mut ds4_bridge_session,
        tokens: *const i32,
        n_tokens: i32,
        progress: ds4_sys::ds4_bridge_prefill_fn,
        ud: *mut c_void,
        _err: *mut c_char,
        _errlen: usize,
    ) -> i32 {
        assert!(!s.is_null());
        assert!(!tokens.is_null());
        match SYNC_FAILURE.with(Cell::get) {
            SyncFailure::Rejected => return 1,
            SyncFailure::Mutated => {
                // Model native forward failure: checkpoint cleared and the
                // generation advanced through the shared mock FFI boundary.
                NATIVE_PREFIX.with(|live| live.borrow_mut().clear());
                ds4_sys::ds4_bridge_session_invalidate(s);
                return 1;
            }
            SyncFailure::None => {}
        }
        let prompt = std::slice::from_raw_parts(tokens, n_tokens as usize);
        let callback = progress.expect("progress callback");

        callback(ud, -1, n_tokens);
        for current in [0, 2, n_tokens] {
            NATIVE_PREFIX.with(|live| {
                *live.borrow_mut() = prompt[..current as usize].to_vec();
            });
            callback(ud, current, n_tokens);
        }
        callback(ud, n_tokens + 1, n_tokens);
        callback(ud, 2, n_tokens + 1);
        SYNC_RC.with(Cell::get)
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_session_save_payload(
        s: *mut ds4_bridge_session,
        path: *const c_char,
        _err: *mut c_char,
        _errlen: usize,
    ) -> i32 {
        assert!(!s.is_null());
        assert!(!path.is_null());
        let path = CStr::from_ptr(path).to_string_lossy();
        let tokens = NATIVE_PREFIX.with(|live| live.borrow().clone());
        let prefix = HostPrefix {
            fields: [
                PAYLOAD_MAGIC,
                PAYLOAD_VERSION,
                4096,
                1,
                4096,
                0,
                0,
                tokens.len() as u32,
                0,
                0,
                0,
                0,
                tokens.len() as u32,
            ],
            tokens: tokens.into_iter().map(|token| token as u32).collect(),
        };
        match std::fs::write(path.as_ref(), prefix.encode()) {
            Ok(()) => 0,
            Err(_) => 1,
        }
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_session_save_layer_payload(
        s: *mut ds4_bridge_session,
        path: *const c_char,
        _layer_start: u32,
        _layer_end: u32,
        _err: *mut c_char,
        _errlen: usize,
    ) -> i32 {
        ds4_bridge_session_save_payload(s, path, _err, _errlen)
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_session_load_layer_payload(
        s: *mut ds4_bridge_session,
        path: *const c_char,
        _payload_bytes: u64,
        _tokens: *const i32,
        _n_tokens: u32,
        _layer_start: u32,
        _layer_end: u32,
        _err: *mut c_char,
        _errlen: usize,
    ) -> i32 {
        assert!(!s.is_null());
        assert!(!path.is_null());
        0
    }

    fn fake_session() -> std::mem::ManuallyDrop<Session<'static>> {
        std::mem::ManuallyDrop::new(Session {
            raw: NonNull::<ds4_bridge_session>::dangling(),
            host: SessionLedger::new(ModelFamily::DeepSeek4, SessionBackend::Cuda, 4096, 4),
            _model: PhantomData,
            _not_send: PhantomData,
        })
    }

    fn iquest_sync_error(failure: SyncFailure) {
        let mut session = fake_session();
        session.host.family = ModelFamily::IQuestQ1;
        session.host.generation = session.native_generation();
        let generation = session.generation();
        let prefix = [1, 2, 3];
        session.host.replace_checkpoint(&prefix);
        session.host.mtp_draft_valid = true;
        NATIVE_PREFIX.with(|live| *live.borrow_mut() = prefix.to_vec());
        SYNC_FAILURE.with(|mode| mode.set(failure));
        let tokens = TokenBuffer::from_tokens(vec![1, 2, 3, 4]);
        let mut callbacks = 0;

        let result = session.sync_progress(&tokens, |_| callbacks += 1);
        SYNC_FAILURE.with(|mode| mode.set(SyncFailure::None));
        assert!(result.is_err());
        assert_eq!(callbacks, 0);
        assert_eq!(session.generation(), session.native_generation());

        if failure == SyncFailure::Rejected {
            assert_eq!(session.generation(), generation);
            assert!(session.host().valid);
            assert!(session.host().mtp_draft_valid);
            assert_eq!(session.host().tokens(), &prefix);
            assert_eq!(session.last_plan(tokens.as_slice()).start, 3);
            return;
        }

        assert_eq!(session.generation(), generation + 1);
        assert!(!session.host().valid);
        assert!(!session.host().mtp_draft_valid);
        assert!(session.host().tokens().is_empty());
        assert_eq!(session.pos(), 0);
        let plan = session.last_plan(tokens.as_slice());
        assert_eq!(plan.start, 0);
        assert!(plan.rebuild);
    }

    #[test]
    fn iquest_sync_failure_ledger() {
        iquest_sync_error(SyncFailure::Mutated);
    }

    #[test]
    fn iquest_sync_reject_keeps() {
        iquest_sync_error(SyncFailure::Rejected);
    }

    #[test]
    fn scopes_exact_prefix() {
        SYNC_RC.with(|rc| rc.set(0));
        let mut session = fake_session();
        let tokens = TokenBuffer::from_tokens(vec![10, 20, 30, 40]);
        let path =
            std::env::temp_dir().join(format!("ds4-prefill-checkpoint-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut seen = Vec::new();

        session
            .sync_progress(&tokens, |checkpoint| {
                seen.push((
                    checkpoint.current(),
                    checkpoint.total(),
                    checkpoint.tokens().to_vec(),
                ));
                if checkpoint.current() == 2 {
                    checkpoint.save_payload(&path).unwrap();
                }
            })
            .unwrap();

        assert_eq!(
            seen,
            vec![
                (0, 4, vec![]),
                (2, 4, vec![10, 20]),
                (4, 4, vec![10, 20, 30, 40]),
            ]
        );
        assert_eq!(session.host().tokens(), tokens.as_slice());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn contains_callback_panic() {
        SYNC_RC.with(|rc| rc.set(0));
        let mut session = fake_session();
        let tokens = TokenBuffer::from_tokens(vec![10, 20, 30, 40]);
        let mut seen = Vec::new();

        session
            .sync_progress(&tokens, |checkpoint| {
                if checkpoint.current() == 2 {
                    panic!("injected progress panic");
                }
                seen.push(checkpoint.current());
            })
            .unwrap();

        assert_eq!(seen, vec![0, 4]);
        assert_eq!(session.host().tokens(), tokens.as_slice());
    }

    #[test]
    fn commits_after_native_success() {
        SYNC_RC.with(|rc| rc.set(9));
        let mut session = fake_session();
        let tokens = TokenBuffer::from_tokens(vec![10, 20, 30, 40]);

        let err = session.sync_progress(&tokens, |_| {}).unwrap_err();

        assert_eq!(err.code, 9);
        assert!(!session.host().valid);
        assert!(session.host().tokens().is_empty());
        SYNC_RC.with(|rc| rc.set(0));
    }
}
