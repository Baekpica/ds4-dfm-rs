//! Continuous batching over the native persistent batch context.
//!
//! The engine's rolling scheduler (mid-flight admit/evict, ragged prefill,
//! per-seq sampling) stays native; the host drives it through a `ContDriver`.
//! All callbacks run on the thread that called `continuous_generate`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use ds4_sys::{
    ds4_bridge_batch_ctx, ds4_bridge_batch_ctx_bank_load_payload_range,
    ds4_bridge_batch_ctx_bank_save_payload, ds4_bridge_batch_ctx_bank_snapshot,
    ds4_bridge_batch_ctx_create_fit, ds4_bridge_batch_ctx_destroy,
    ds4_bridge_batch_ctx_generate_static, ds4_bridge_batch_ctx_max_seq,
    ds4_bridge_batch_ctx_raw_cap, ds4_bridge_batch_ctx_seq_cap,
    ds4_bridge_batch_ctx_supports_partial_reuse, ds4_bridge_batch_ctx_trim_free,
    ds4_bridge_cont_request, ds4_bridge_cont_stats, ds4_bridge_continuous_generate,
    ds4_bridge_qwen_image_info, ds4_bridge_qwen_image_input, ds4_bridge_qwen_image_pixel_hash,
    ds4_bridge_qwen_image_probe,
};

use crate::{cstring_payload_path, fail, Error, Model, Result};

/// `ds4_cont_request.sample_override` result encoding (`DS4_SAMPLE_OVERRIDE_*`).
pub const CONT_SAMPLE_NONE: i32 = 0;
pub const CONT_SAMPLE_GREEDY: i32 = 1;

pub fn cont_sample_token(token_id: i32) -> i32 {
    token_id + 2
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QwenImageInfo {
    pub source_width: u32,
    pub source_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub grid_h: u32,
    pub grid_w: u32,
    pub token_count: u32,
}

#[derive(Clone, Debug)]
pub struct QwenImageInput {
    pub data: Arc<[u8]>,
    pub token_offset: u32,
    pub grid_h: u32,
    pub grid_w: u32,
}

pub fn qwen_image_probe(data: &[u8]) -> Result<QwenImageInfo> {
    let mut info = ds4_bridge_qwen_image_info::default();
    let mut err = [0u8; 512];
    let rc = unsafe {
        ds4_bridge_qwen_image_probe(
            data.as_ptr(),
            data.len(),
            &mut info,
            err.as_mut_ptr() as *mut c_char,
            err.len(),
        )
    };
    if rc != 0 {
        return Err(fail(rc, &err));
    }
    Ok(QwenImageInfo {
        source_width: info.source_width,
        source_height: info.source_height,
        resized_width: info.resized_width,
        resized_height: info.resized_height,
        grid_h: info.grid_h,
        grid_w: info.grid_w,
        token_count: info.token_count,
    })
}

pub fn qwen_image_pixel_hash(data: &[u8]) -> Result<u64> {
    let mut hash = 0;
    let mut err = [0u8; 512];
    let rc = unsafe {
        ds4_bridge_qwen_image_pixel_hash(
            data.as_ptr(),
            data.len(),
            &mut hash,
            err.as_mut_ptr() as *mut c_char,
            err.len(),
        )
    };
    if rc != 0 {
        return Err(fail(rc, &err));
    }
    Ok(hash)
}

/// One admission. `tokens` moves into the batch context and stays alive
/// until that request's `on_done` fires (the engine borrows the buffer).
#[derive(Debug, Clone)]
pub struct ContAdmit {
    pub user: usize,
    pub tokens: Vec<i32>,
    pub images: Vec<QwenImageInput>,
    pub max_new: i32,
    /// `< 0` selects the engine/family default EOS.
    pub eos: i32,
    /// `<= 0` is greedy argmax (ignores the rest of the sampling block).
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    /// Bank id + 1 placement directive; 0 = engine's choice.
    pub place_bank: i32,
    /// Committed prefix length for a warm admit; 0 = cold.
    pub n_cached: i32,
    /// Source bank id + 1 for fork-by-copy; 0 = no fork.
    pub fork_bank: i32,
}

impl ContAdmit {
    pub fn cold(user: usize, tokens: Vec<i32>, max_new: i32) -> Self {
        Self {
            user,
            tokens,
            images: Vec::new(),
            max_new,
            eos: -1,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            min_p: 0.0,
            seed: 0,
            place_bank: 0,
            n_cached: 0,
            fork_bank: 0,
        }
    }
}

/// Host half of `ds4_engine_continuous_generate`. Same contracts as the C
/// callbacks: `admit` returning `None` plus an empty active set ends the
/// loop; `on_token(false)` aborts that sequence; `on_admitted(false)`
/// cancels before prefill; `alive(false)` abandons a pending admission.
pub trait ContDriver {
    fn admit(&mut self) -> Option<ContAdmit>;
    fn on_token(&mut self, user: usize, token: i32) -> bool;
    fn on_done(
        &mut self,
        user: usize,
        tokens: &[i32],
        finish: i32,
        decode_ms: f64,
        decode_tokens: i32,
        decode_steps: i32,
    );
    fn sample_override(&mut self, _user: usize) -> i32 {
        CONT_SAMPLE_NONE
    }
    fn alive(&mut self, _user: usize) -> bool {
        true
    }
    fn on_admitted(&mut self, _user: usize, _n_cached: i32, _n_computed: i32, _bank: i32) -> bool {
        true
    }
}

struct LiveAdmit {
    tokens: Vec<i32>,
    images: Vec<QwenImageInput>,
}

struct TrampCtx<'a> {
    driver: &'a mut dyn ContDriver,
    /// Prompt/image buffers the engine may still read; freed on that user's done.
    live: HashMap<usize, LiveAdmit>,
}

unsafe extern "C" fn tramp_admit(ud: *mut c_void, req: *mut ds4_bridge_cont_request) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    let admitted = catch_unwind(AssertUnwindSafe(|| t.driver.admit())).unwrap_or(None);
    let Some(a) = admitted else { return 0 };
    if a.images.len() > 4 {
        return 0;
    }
    let user = a.user;
    let entry = t.live.entry(user).or_insert_with(|| LiveAdmit {
        tokens: Vec::new(),
        images: Vec::new(),
    });
    entry.tokens = a.tokens;
    entry.images = a.images;
    let r = &mut *req;
    r.tokens = entry.tokens.as_ptr();
    r.n = entry.tokens.len() as i32;
    for (dst, image) in r.images.iter_mut().zip(&entry.images) {
        *dst = ds4_bridge_qwen_image_input {
            data: image.data.as_ptr(),
            data_len: image.data.len(),
            token_offset: image.token_offset,
            grid_h: image.grid_h,
            grid_w: image.grid_w,
        };
    }
    r.image_count = entry.images.len() as u32;
    r.max_new = a.max_new;
    r.eos = a.eos;
    r.user = user as *mut c_void;
    r.temperature = a.temperature;
    r.top_k = a.top_k;
    r.top_p = a.top_p;
    r.min_p = a.min_p;
    r.seed = a.seed;
    r.sample_override = Some(tramp_sample_override);
    r.alive = Some(tramp_alive);
    r.on_admitted = Some(tramp_on_admitted);
    r.place_bank = a.place_bank;
    r.n_cached = a.n_cached;
    r.bank_used = ptr::null_mut();
    r.fork_bank = a.fork_bank;
    1
}

unsafe extern "C" fn tramp_on_token(ud: *mut c_void, user: *mut c_void, token: i32) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    let cont =
        catch_unwind(AssertUnwindSafe(|| t.driver.on_token(user as usize, token))).unwrap_or(false);
    i32::from(cont)
}

unsafe extern "C" fn tramp_on_done(
    ud: *mut c_void,
    user: *mut c_void,
    tokens: *const i32,
    n: i32,
    finish: i32,
    stats: *const ds4_bridge_cont_stats,
) {
    let t = &mut *(ud as *mut TrampCtx);
    let user = user as usize;
    let toks: &[i32] = if tokens.is_null() || n <= 0 {
        &[]
    } else {
        std::slice::from_raw_parts(tokens, n as usize)
    };
    let stats = stats.as_ref().copied().unwrap_or_default();
    let _ = catch_unwind(AssertUnwindSafe(|| {
        t.driver.on_done(
            user,
            toks,
            finish,
            stats.decode_ms,
            stats.decode_tokens as i32,
            stats.decode_steps as i32,
        )
    }));
    t.live.remove(&user);
}

unsafe extern "C" fn tramp_sample_override(ud: *mut c_void, user: *mut c_void) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    catch_unwind(AssertUnwindSafe(|| t.driver.sample_override(user as usize)))
        .unwrap_or(CONT_SAMPLE_NONE)
}

unsafe extern "C" fn tramp_alive(ud: *mut c_void, user: *mut c_void) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    i32::from(catch_unwind(AssertUnwindSafe(|| t.driver.alive(user as usize))).unwrap_or(true))
}

unsafe extern "C" fn tramp_on_admitted(
    ud: *mut c_void,
    user: *mut c_void,
    n_cached: c_int,
    n_computed: c_int,
    bank: c_int,
) -> c_int {
    let t = &mut *(ud as *mut TrampCtx);
    i32::from(
        catch_unwind(AssertUnwindSafe(|| {
            t.driver
                .on_admitted(user as usize, n_cached, n_computed, bank)
        }))
        .unwrap_or(true),
    )
}

pub struct BatchCtx<'m> {
    raw: NonNull<ds4_bridge_batch_ctx>,
    _model: PhantomData<&'m Model>,
    _not_send: PhantomData<*const ()>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankSnapshot {
    pub bank: i32,
    pub tokens: Vec<i32>,
    pub generation: u64,
}

/// One greedy request for [`BatchCtx::generate_static`].  The native engine
/// borrows `tokens` only for the duration of the call.  A non-positive budget
/// keeps the C contract and generates at most one token.
#[derive(Clone, Copy, Debug)]
pub struct StaticBatchRequest<'a> {
    pub tokens: &'a [i32],
    pub max_new_tokens: i32,
    /// `< 0` selects the engine/family default EOS.
    pub eos: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticBatchFinish {
    Budget,
    Eos,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticBatchResult {
    pub tokens: Vec<i32>,
    pub finish: StaticBatchFinish,
}

impl Model {
    /// `ds4_batch_ctx_create_fit`: `max_seq` is a cap; the engine sizes the
    /// bank count down to the memory budget. Read the width back with
    /// [`BatchCtx::max_seq`].
    pub fn batch_ctx_fit(
        &self,
        ctx_size: i32,
        max_seq: i32,
        max_total_tokens: i32,
    ) -> Result<BatchCtx<'_>> {
        let mut raw = ptr::null_mut();
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_batch_ctx_create_fit(
                self.raw_ptr(),
                ctx_size,
                max_seq,
                max_total_tokens,
                &mut raw,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(crate::fail(rc, &err));
        }
        let raw = NonNull::new(raw).ok_or_else(|| Error {
            code: 1,
            message: "ds4_bridge_batch_ctx_create_fit returned NULL".into(),
        })?;
        Ok(BatchCtx {
            raw,
            _model: PhantomData,
            _not_send: PhantomData,
        })
    }
}

impl BatchCtx<'_> {
    pub fn max_seq(&self) -> i32 {
        unsafe { ds4_bridge_batch_ctx_max_seq(self.raw.as_ptr()) }
    }

    pub fn seq_cap(&self) -> i32 {
        unsafe { ds4_bridge_batch_ctx_seq_cap(self.raw.as_ptr()) }
    }

    pub fn supports_partial_reuse(&self) -> bool {
        unsafe { ds4_bridge_batch_ctx_supports_partial_reuse(self.raw.as_ptr()) != 0 }
    }

    pub fn trim_free(&mut self, want_bytes: u64) -> u64 {
        unsafe { ds4_bridge_batch_ctx_trim_free(self.raw.as_ptr(), want_bytes) }
    }

    /// Runs one fixed group to completion on the persistent native context.
    /// Native result allocations are copied and released inside the bridge;
    /// every returned token buffer is owned by Rust.
    pub fn generate_static(
        &mut self,
        requests: &[StaticBatchRequest<'_>],
    ) -> Result<Vec<StaticBatchResult>> {
        if requests.is_empty() {
            return Err(Error {
                code: 1,
                message: "static batch requires at least one request".into(),
            });
        }
        let max_seq = self.max_seq();
        if max_seq <= 0 || requests.len() > max_seq as usize {
            return Err(Error {
                code: 1,
                message: format!(
                    "static batch request count {} exceeds context width {}",
                    requests.len(),
                    max_seq
                ),
            });
        }
        let static_token_cap = unsafe { ds4_bridge_batch_ctx_raw_cap(self.raw.as_ptr()) };
        if static_token_cap <= 0 {
            return Err(Error {
                code: 1,
                message: "static batch context has no raw token capacity".into(),
            });
        }

        let mut prompt_tokens = Vec::with_capacity(requests.len());
        let mut prompt_lengths = Vec::with_capacity(requests.len());
        let mut max_new_tokens = Vec::with_capacity(requests.len());
        let mut eos_ids = Vec::with_capacity(requests.len());
        let mut output_cap = 0usize;
        for (index, request) in requests.iter().enumerate() {
            if request.tokens.is_empty() {
                return Err(Error {
                    code: 1,
                    message: format!("static batch prompt {index} is empty"),
                });
            }
            let prompt_len = i32::try_from(request.tokens.len()).map_err(|_| Error {
                code: 1,
                message: format!("static batch prompt {index} is too large"),
            })?;
            let requested_cap = if request.max_new_tokens > 0 {
                request.max_new_tokens as usize
            } else {
                1
            };
            let context_room = if static_token_cap > prompt_len {
                (static_token_cap - prompt_len) as usize
            } else {
                1
            };
            let request_cap = requested_cap.min(context_room);
            output_cap = output_cap.checked_add(request_cap).ok_or_else(|| Error {
                code: 1,
                message: "static batch output capacity overflows".into(),
            })?;
            prompt_tokens.push(request.tokens.as_ptr());
            prompt_lengths.push(prompt_len);
            max_new_tokens.push(request.max_new_tokens);
            eos_ids.push(request.eos);
        }
        let output_cap_i32 = i32::try_from(output_cap).map_err(|_| Error {
            code: 1,
            message: "static batch output capacity exceeds native ABI".into(),
        })?;
        let n = requests.len() as i32;
        let mut flat_tokens = Vec::new();
        flat_tokens
            .try_reserve_exact(output_cap)
            .map_err(|_| Error {
                code: 1,
                message: "failed to allocate static batch output".into(),
            })?;
        flat_tokens.resize(output_cap, 0);
        let mut output_lengths = vec![0; requests.len()];
        let mut output_finish = vec![0; requests.len()];
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_batch_ctx_generate_static(
                self.raw.as_ptr(),
                prompt_tokens.as_ptr(),
                prompt_lengths.as_ptr(),
                max_new_tokens.as_ptr(),
                eos_ids.as_ptr(),
                n,
                flat_tokens.as_mut_ptr(),
                output_cap_i32,
                output_lengths.as_mut_ptr(),
                output_finish.as_mut_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }

        let mut cursor = 0usize;
        let mut results = Vec::with_capacity(requests.len());
        for (index, (&length, &finish)) in
            output_lengths.iter().zip(output_finish.iter()).enumerate()
        {
            let length = usize::try_from(length).map_err(|_| Error {
                code: 1,
                message: format!("invalid native static batch length at {index}"),
            })?;
            let end = cursor.checked_add(length).ok_or_else(|| Error {
                code: 1,
                message: "native static batch length overflow".into(),
            })?;
            if end > flat_tokens.len() {
                return Err(Error {
                    code: 1,
                    message: format!("native static batch output {index} exceeds its buffer"),
                });
            }
            let finish = match finish {
                0 => StaticBatchFinish::Budget,
                1 => StaticBatchFinish::Eos,
                _ => {
                    return Err(Error {
                        code: 1,
                        message: format!("invalid native static batch finish at {index}"),
                    })
                }
            };
            results.push(StaticBatchResult {
                tokens: flat_tokens[cursor..end].to_vec(),
                finish,
            });
            cursor = end;
        }
        Ok(results)
    }

    /// Copies the native committed frontier; no C token pointer escapes.
    pub fn bank_snapshot(&self, bank: i32) -> Result<BankSnapshot> {
        let cap = self.seq_cap();
        if cap <= 0 {
            return Err(Error {
                code: 1,
                message: "batch context has no token capacity".into(),
            });
        }
        let mut tokens = vec![0; cap as usize];
        let mut n = 0;
        let mut generation = 0;
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_batch_ctx_bank_snapshot(
                self.raw.as_ptr(),
                bank,
                tokens.as_mut_ptr(),
                cap,
                &mut n,
                &mut generation,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        if n < 0 || n > cap {
            return Err(Error {
                code: 1,
                message: format!("invalid native bank snapshot bank={bank} tokens={n} cap={cap}"),
            });
        }
        tokens.truncate(n as usize);
        Ok(BankSnapshot {
            bank,
            tokens,
            generation,
        })
    }

    pub fn save_bank_payload(&self, bank: i32, path: impl AsRef<Path>) -> Result<()> {
        let c_path = cstring_payload_path(path.as_ref())?;
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_batch_ctx_bank_save_payload(
                self.raw.as_ptr(),
                bank,
                c_path.as_ptr(),
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        Ok(())
    }

    pub fn load_bank_payload_range(
        &self,
        bank: i32,
        path: impl AsRef<Path>,
        offset: u64,
        length: u64,
    ) -> Result<BankSnapshot> {
        let c_path = cstring_payload_path(path.as_ref())?;
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_batch_ctx_bank_load_payload_range(
                self.raw.as_ptr(),
                bank,
                c_path.as_ptr(),
                offset,
                length,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(fail(rc, &err));
        }
        self.bank_snapshot(bank)
    }

    /// Runs the engine's rolling loop until the active set is empty and
    /// `driver.admit()` returns `None`.
    pub fn continuous_generate(&self, driver: &mut dyn ContDriver) -> Result<()> {
        let mut t = TrampCtx {
            driver,
            live: HashMap::new(),
        };
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_continuous_generate(
                self.raw.as_ptr(),
                Some(tramp_admit),
                Some(tramp_on_token),
                Some(tramp_on_done),
                &mut t as *mut TrampCtx as *mut c_void,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(crate::fail(rc, &err));
        }
        let _ = &t;
        Ok(())
    }
}

impl Drop for BatchCtx<'_> {
    fn drop(&mut self) {
        unsafe { ds4_bridge_batch_ctx_destroy(self.raw.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::ffi::CStr;
    use std::mem::ManuallyDrop;

    thread_local! {
        static TOKENS: RefCell<Vec<i32>> = RefCell::new(vec![10, 20, 30]);
        static LOAD: RefCell<Option<(i32, String, u64, u64)>> = const { RefCell::new(None) };
        static SAVE_BANK: Cell<i32> = const { Cell::new(-1) };
        static STATIC_INPUT: RefCell<Vec<(Vec<i32>, i32, i32)>> = const { RefCell::new(Vec::new()) };
        static STATIC_OUTPUT_CAP: Cell<i32> = const { Cell::new(0) };
        static CONT_INPUT: RefCell<Option<(Vec<i32>, Vec<u8>, u32, u32, u32)>> = const { RefCell::new(None) };
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_batch_ctx_generate_static(
        _c: *mut ds4_bridge_batch_ctx,
        prompt_tokens: *const *const i32,
        prompt_lengths: *const i32,
        max_new_tokens: *const i32,
        eos_ids: *const i32,
        n: i32,
        out_tokens: *mut i32,
        out_tokens_cap: i32,
        out_lengths: *mut i32,
        out_finish: *mut i32,
        _err: *mut c_char,
        _errlen: usize,
    ) -> c_int {
        let mut seen = Vec::new();
        for i in 0..n as usize {
            let len = *prompt_lengths.add(i) as usize;
            let tokens = std::slice::from_raw_parts(*prompt_tokens.add(i), len).to_vec();
            seen.push((tokens, *max_new_tokens.add(i), *eos_ids.add(i)));
        }
        STATIC_INPUT.with(|input| *input.borrow_mut() = seen);
        STATIC_OUTPUT_CAP.with(|cap| cap.set(out_tokens_cap));
        if out_tokens_cap < 3 {
            return 1;
        }
        std::ptr::copy_nonoverlapping([101, 102, 201].as_ptr(), out_tokens, 3);
        std::ptr::copy_nonoverlapping([2, 1].as_ptr(), out_lengths, 2);
        std::ptr::copy_nonoverlapping([1, 0].as_ptr(), out_finish, 2);
        0
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_qwen_image_probe(
        _data: *const u8,
        data_len: usize,
        info: *mut ds4_bridge_qwen_image_info,
        _err: *mut c_char,
        _errlen: usize,
    ) -> c_int {
        if data_len != 3 || info.is_null() {
            return 1;
        }
        *info = ds4_bridge_qwen_image_info {
            source_width: 1,
            source_height: 2,
            resized_width: 256,
            resized_height: 512,
            grid_h: 32,
            grid_w: 16,
            token_count: 128,
        };
        0
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_qwen_image_pixel_hash(
        _data: *const u8,
        data_len: usize,
        hash: *mut u64,
        _err: *mut c_char,
        _errlen: usize,
    ) -> c_int {
        if data_len != 3 || hash.is_null() {
            return 1;
        }
        *hash = 0x0123_4567_89ab_cdef;
        0
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_continuous_generate(
        _c: *mut ds4_bridge_batch_ctx,
        admit: Option<unsafe extern "C" fn(*mut c_void, *mut ds4_bridge_cont_request) -> c_int>,
        _on_token: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> c_int>,
        on_done: Option<
            unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const i32,
                i32,
                i32,
                *const ds4_bridge_cont_stats,
            ),
        >,
        ud: *mut c_void,
        _err: *mut c_char,
        _errlen: usize,
    ) -> c_int {
        loop {
            let mut req: ds4_bridge_cont_request = std::mem::zeroed();
            if admit.is_none_or(|admit| admit(ud, &mut req) == 0) {
                break;
            }
            let tokens = std::slice::from_raw_parts(req.tokens, req.n as usize).to_vec();
            if req.image_count > 0 {
                let image = &req.images[0];
                let data = std::slice::from_raw_parts(image.data, image.data_len).to_vec();
                CONT_INPUT.with(|input| {
                    *input.borrow_mut() =
                        Some((tokens, data, image.token_offset, image.grid_h, image.grid_w));
                });
            }
            if let Some(on_admitted) = req.on_admitted {
                on_admitted(ud, req.user, req.n_cached, req.n - req.n_cached, 0);
            }
            if let Some(on_done) = on_done {
                on_done(
                    ud,
                    req.user,
                    std::ptr::null(),
                    0,
                    1,
                    &ds4_bridge_cont_stats::default(),
                );
            }
        }
        0
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_batch_ctx_max_seq(_c: *mut ds4_bridge_batch_ctx) -> c_int {
        8
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_batch_ctx_raw_cap(_c: *mut ds4_bridge_batch_ctx) -> c_int {
        8
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_batch_ctx_seq_cap(_c: *mut ds4_bridge_batch_ctx) -> c_int {
        10
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_batch_ctx_supports_partial_reuse(
        _c: *mut ds4_bridge_batch_ctx,
    ) -> c_int {
        1
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_batch_ctx_trim_free(
        _c: *mut ds4_bridge_batch_ctx,
        want_bytes: u64,
    ) -> u64 {
        want_bytes
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_batch_ctx_bank_snapshot(
        _c: *mut ds4_bridge_batch_ctx,
        bank: i32,
        out: *mut i32,
        cap: i32,
        n: *mut i32,
        generation: *mut u64,
        _err: *mut c_char,
        _errlen: usize,
    ) -> c_int {
        let tokens = TOKENS.with(|tokens| tokens.borrow().clone());
        if cap < tokens.len() as i32 {
            *n = tokens.len() as i32;
            return 1;
        }
        std::ptr::copy_nonoverlapping(tokens.as_ptr(), out, tokens.len());
        *n = tokens.len() as i32;
        *generation = 41 + bank as u64;
        0
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_batch_ctx_bank_save_payload(
        _c: *mut ds4_bridge_batch_ctx,
        bank: i32,
        path: *const c_char,
        _err: *mut c_char,
        _errlen: usize,
    ) -> c_int {
        SAVE_BANK.with(|seen| seen.set(bank));
        std::fs::write(CStr::from_ptr(path).to_string_lossy().as_ref(), b"BANK").is_err() as i32
    }

    #[no_mangle]
    unsafe extern "C" fn ds4_bridge_batch_ctx_bank_load_payload_range(
        _c: *mut ds4_bridge_batch_ctx,
        bank: i32,
        path: *const c_char,
        offset: u64,
        length: u64,
        _err: *mut c_char,
        _errlen: usize,
    ) -> c_int {
        LOAD.with(|load| {
            *load.borrow_mut() = Some((
                bank,
                CStr::from_ptr(path).to_string_lossy().into_owned(),
                offset,
                length,
            ));
        });
        0
    }

    fn fake_batch() -> ManuallyDrop<BatchCtx<'static>> {
        ManuallyDrop::new(BatchCtx {
            raw: NonNull::dangling(),
            _model: PhantomData,
            _not_send: PhantomData,
        })
    }

    #[test]
    fn bank_snapshot_is_an_owned_copy() {
        TOKENS.with(|tokens| *tokens.borrow_mut() = vec![10, 20, 30]);
        let batch = fake_batch();
        let snapshot = batch.bank_snapshot(2).unwrap();
        TOKENS.with(|tokens| tokens.borrow_mut()[0] = 99);

        assert_eq!(snapshot.tokens, [10, 20, 30]);
        assert_eq!(snapshot.generation, 43);
    }

    #[test]
    fn qwen_image_probe_and_hash_hide_the_native_layout() {
        let data = [1, 2, 3];
        assert_eq!(
            qwen_image_probe(&data).unwrap(),
            QwenImageInfo {
                source_width: 1,
                source_height: 2,
                resized_width: 256,
                resized_height: 512,
                grid_h: 32,
                grid_w: 16,
                token_count: 128,
            }
        );
        assert_eq!(qwen_image_pixel_hash(&data).unwrap(), 0x0123_4567_89ab_cdef);
    }

    #[test]
    fn continuous_admit_keeps_qwen_image_payload_alive() {
        struct Driver {
            pending: bool,
            done: bool,
        }
        impl ContDriver for Driver {
            fn admit(&mut self) -> Option<ContAdmit> {
                if !self.pending {
                    return None;
                }
                self.pending = false;
                let mut admit = ContAdmit::cold(7, vec![10, 20], 1);
                admit.images.push(QwenImageInput {
                    data: vec![1, 2, 3].into(),
                    token_offset: 11,
                    grid_h: 32,
                    grid_w: 16,
                });
                Some(admit)
            }

            fn on_token(&mut self, _user: usize, _token: i32) -> bool {
                true
            }

            fn on_done(
                &mut self,
                user: usize,
                _tokens: &[i32],
                _finish: i32,
                _decode_ms: f64,
                _decode_tokens: i32,
                _decode_steps: i32,
            ) {
                assert_eq!(user, 7);
                self.done = true;
            }
        }

        CONT_INPUT.with(|input| *input.borrow_mut() = None);
        let mut driver = Driver {
            pending: true,
            done: false,
        };
        fake_batch().continuous_generate(&mut driver).unwrap();
        assert!(driver.done);
        assert_eq!(
            CONT_INPUT.with(|input| input.borrow().clone()).unwrap(),
            (vec![10, 20], vec![1, 2, 3], 11, 32, 16)
        );
    }

    #[test]
    fn continuous_driver_is_repolled_after_each_completed_slot() {
        struct Driver {
            next: usize,
            admitted: Vec<usize>,
            done: Vec<usize>,
        }
        impl ContDriver for Driver {
            fn admit(&mut self) -> Option<ContAdmit> {
                if self.next == 3 {
                    return None;
                }
                self.next += 1;
                Some(ContAdmit::cold(self.next, vec![10], 1))
            }

            fn on_token(&mut self, _user: usize, _token: i32) -> bool {
                true
            }

            fn on_done(
                &mut self,
                user: usize,
                _tokens: &[i32],
                _finish: i32,
                _decode_ms: f64,
                _decode_tokens: i32,
                _decode_steps: i32,
            ) {
                self.done.push(user);
            }

            fn on_admitted(
                &mut self,
                user: usize,
                _n_cached: i32,
                _n_computed: i32,
                bank: i32,
            ) -> bool {
                assert_eq!(bank, 0);
                self.admitted.push(user);
                true
            }
        }

        let mut driver = Driver {
            next: 0,
            admitted: Vec::new(),
            done: Vec::new(),
        };
        fake_batch().continuous_generate(&mut driver).unwrap();
        assert_eq!(driver.admitted, [1, 2, 3]);
        assert_eq!(driver.done, [1, 2, 3]);
    }

    #[test]
    fn partial_reuse_capability_is_read_from_the_native_batch() {
        assert!(fake_batch().supports_partial_reuse());
    }

    #[test]
    fn trim_free_forwards_want_to_the_native_batch() {
        let mut batch = fake_batch();
        assert_eq!(batch.trim_free(0), 0);
        assert_eq!(batch.trim_free(4096), 4096);
    }

    #[test]
    fn bank_payload_paths_stay_opaque() {
        TOKENS.with(|tokens| *tokens.borrow_mut() = vec![10, 20, 30]);
        let batch = fake_batch();
        let path =
            std::env::temp_dir().join(format!("ds4-core-bank-payload-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);

        batch.save_bank_payload(1, &path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"BANK");
        assert_eq!(SAVE_BANK.with(Cell::get), 1);
        let restored = batch.load_bank_payload_range(1, &path, 7, 11).unwrap();
        let load = LOAD.with(|load| load.borrow().clone()).unwrap();
        assert_eq!((load.0, load.2, load.3), (1, 7, 11));
        assert_eq!(restored.tokens, [10, 20, 30]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn static_batch_results_are_owned_and_hide_the_native_layout() {
        let batch = &mut *fake_batch();
        let first = vec![10, 20];
        let second = vec![30];
        let requests = [
            StaticBatchRequest {
                tokens: &first,
                max_new_tokens: 8,
                eos: 99,
            },
            StaticBatchRequest {
                tokens: &second,
                max_new_tokens: 1,
                eos: -1,
            },
        ];

        let results = batch.generate_static(&requests).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].tokens, [101, 102]);
        assert_eq!(results[0].finish, StaticBatchFinish::Eos);
        assert_eq!(results[1].tokens, [201]);
        assert_eq!(results[1].finish, StaticBatchFinish::Budget);
        assert_eq!(STATIC_OUTPUT_CAP.with(Cell::get), 7);
        assert_eq!(
            STATIC_INPUT.with(|input| input.borrow().clone()),
            vec![(vec![10, 20], 8, 99), (vec![30], 1, -1)]
        );
    }
}
