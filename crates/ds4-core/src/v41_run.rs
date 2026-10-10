//! V4.1 (ds41) run surface (P5): the Rust host's side of the engine's
//! single-request generate.
//!
//! The engine's server path IS the one-shot generate with two callbacks
//! (`server_generate_v41.c:412-414` sets the prefill progress hook and calls
//! `ds4_engine_v41_generate_argmax` with `v41_emit`), so the port keeps that
//! shape: one bridge entry, an emit closure per produced token (returning
//! false stops, the CLI's EOS convention), a progress closure per prefill
//! chunk (returning false aborts).
//!
//! Engram rows are host work by design (P4-4): the hash and the shard preads
//! live here (`engram.rs`, verified against the golden erows), the pinned
//! buffers stay native, and the native hands the provider the exact block the
//! forward is about to run (`ds41_forward.h`).

use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use ds4_sys::{
    ds4_bridge_v41_generate, ds4_bridge_v41_last_spec_stats, ds4_bridge_v41_set_dspark,
    ds4_bridge_v41_set_emit_trace, ds4_bridge_v41_set_graph, ds4_bridge_v41_set_prof,
    ds4_bridge_v41_set_sampling, ds4_bridge_v41_zchain_gr_set, ds4_bridge_v41_zchain_rb_clear,
    ds4_bridge_v41_zchain_rb_set,
};

use crate::engram::{EngramHash, EngramShard};
use crate::uring::Req;
use crate::{
    identify_file, Error, GgufFile, Model, Result, TensorInventory, V41Wire, V41Zchain, ZchainGeom,
};

/// `--mem-budget-mb` (unit Z): the engine's process memory red line and L1
/// load gate (`core_util.c:198-256`).  Call before opening the model; 0
/// leaves the budget unset (the gate is then a no-op, the engine's own
/// default).  On Linux the footprint-watchdog half is inert by the engine's
/// own design (`core_util.c:141-145`); the L1 gate is the active half.
pub fn set_mem_budget_mb(mb: i32) {
    unsafe {
        ds4_sys::ds4_bridge_set_mem_budget_mb(mb);
    }
}

/// The last V4.1 run's speculation account: (rounds, offered, accepted), the
/// engine's `ds4_engine_v41_last_spec_stats` (`core_v41_api.c:41-46`; the
/// counters are cleared by the next generate entry).  The server reports the
/// rates after the call returns.
pub fn v41_last_spec_stats() -> (i32, i32, i32) {
    let mut rounds = 0i32;
    let mut offered = 0i32;
    let mut accepted = 0i32;
    unsafe {
        ds4_bridge_v41_last_spec_stats(&mut rounds, &mut offered, &mut accepted);
    }
    (rounds, offered, accepted)
}

/// The decode-sampling face for one V4.1 run (the engine's
/// `ds4_decode_sampling`, `ds4_v41_api.h:69-75`): temperature > 0 with no
/// penalties takes the device sampling kernel (speculation runs, rejection
/// sampling in-kernel); any non-zero penalty takes the host penalty route
/// (the V4 sampler after freq / presence / DRY rewrite the row; temperature 0
/// yields argmax there); all-zero is bare argmax.  `seed` 0 = clock (the V4
/// CLI's rule, `core_v41_api.c:163-164`).  `dry_base` / `dry_allowed_length`
/// matter only when `dry_multiplier` > 0.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V41Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: i32,
    pub seed: u64,
    pub freq_penalty: f32,
    pub presence_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
}

/// The per-run switches (the engine's CLI globals, `cli_diag.c:58`).
#[derive(Clone, Debug)]
pub struct V41RunOptions<'a> {
    /// `--engram-dir`: the converter wrote the converting machine's absolute
    /// shard paths; the directory is swapped, the file name kept
    /// (`core_validate_v41.c:11-13`).
    pub engram_dir: Option<&'a str>,
    /// `--v41-no-engram` (diagnostic): skip the engram layers entirely.
    pub no_engram: bool,
    /// `--dspark` = Some(2), `--no-dspark` = Some(0); None leaves the native
    /// default (1, or the DS41_NO_DSPARK env fallback).
    pub dspark: Option<i32>,
    /// `--no-graph` = Some(false); None leaves the native default.
    pub graph: Option<bool>,
    /// `--dspark-verify K`: pin the round's k; 0 = the confidence scheduler.
    pub verify_k: i32,
    /// `--emit-trace`: the per-round `[dspark]` lines.
    pub emit_trace: bool,
    /// `--v41-prof`: per-layer ms and the full `[dspark]` lines.
    pub prof: bool,
    /// The per-request sampling face (`ds4_engine_set_decode_sampling`,
    /// `core_v41_api.c:10-13`).  Some = set it before the run (the engine's
    /// server and CLI always do, `server_generate_v41.c:409` /
    /// `cli_diag.c:50`); None resets it to the all-zero face = bare argmax,
    /// so a run never inherits the previous request's face.
    pub sampling: Option<V41Sampling>,
    /// `--zchain <dir>`: the ② anti-repair sidecar (install.sh's own
    /// spelling; the deployment's third file next to the GGUF).  The engine
    /// carries it to the per-state mount (`core_engine_open.c:120-124` ->
    /// `v41_state_plugins`, `core_v41_state.c:77-87`); the port mounts it
    /// here, before the native call, with the same files read on the Rust
    /// host (`zchain.rs`, P2/P3).
    pub zchain: Option<&'a str>,
    /// `--posttrain <dir>`: the ③ post-train gain directory (the
    /// `base.fnv`-gated third file; finance-only in the deployment,
    /// install.sh).  Merged with ② exactly as `core_v41_amp.c:227-270`.
    pub posttrain: Option<&'a str>,
}

impl Default for V41RunOptions<'_> {
    fn default() -> Self {
        Self {
            engram_dir: None,
            no_engram: false,
            dspark: None,
            graph: None,
            verify_k: 0,
            emit_trace: false,
            prof: false,
            sampling: None,
            zchain: None,
            posttrain: None,
        }
    }
}

/// The engram rows provider: the hash tables and one open shard per engram
/// layer, plus the token history the n-gram window reads.
pub struct V41Feed {
    hash: EngramHash,
    shards: Vec<EngramShard>,
    weight_off: Vec<u64>,
    scale_off: Vec<u64>,
    rows: Vec<u64>,
    head_dim: u32,
    cols: u32,
    stride: u32,
    hist: Vec<i32>,
    /// The engine's io_uring channel (`v41_ering`, core_v41_ering.c): one
    /// submit per engram layer's round; None keeps the serial preads.
    ring: Option<crate::uring::Ring>,
}

impl V41Feed {
    /// Open the feed from the artifact: the wire (with the `--engram-dir`
    /// rewrite), the four hash-constant tensors, and the shards.  None when
    /// the model carries no engram layers (the forward then takes no feed).
    pub fn open(path: &Path, engram_dir: Option<&str>) -> Result<Option<Self>> {
        let g = GgufFile::open(path).map_err(asset_error)?;
        let inv = TensorInventory::open(path).map_err(asset_error)?;
        let identified = identify_file(&g).map_err(asset_error)?;
        let mut wire = V41Wire::load(&g, &identified.shape).map_err(asset_error)?;
        if let Some(dir) = engram_dir {
            wire.apply_engram_dir(dir);
        }
        let Some(hash) = EngramHash::load(&g, &inv, &wire, identified.shape.n_vocab)
            .map_err(asset_error)?
        else {
            return Ok(None);
        };
        let mut shards = Vec::with_capacity(wire.engram_layers.len());
        for path in &wire.engram_table_path {
            let shard = EngramShard::open(Path::new(path)).map_err(asset_error)?;
            shards.push(shard);
        }
        let head_dim = wire.engram_head_dim;
        let stride = head_dim + head_dim / 32;
        // The engine's row channel (core_v41_engram.c:231-233): io_uring when
        // the kernel takes it, serial preads otherwise.  Its own line, so a
        // gate log shows which path ran.
        let ring = crate::uring::Ring::open(crate::uring::ERING_CAP);
        match &ring {
            Some(r) => eprintln!(
                "ds4: [v41] engram rows: io_uring (one submit per round, depth {})",
                r.cap()
            ),
            None => eprintln!("ds4: [v41] engram rows: serial preads (no io_uring)"),
        }
        Ok(Some(Self {
            hash,
            shards,
            weight_off: wire.engram_weight_off.clone(),
            scale_off: wire.engram_scale_off.clone(),
            rows: wire.engram_rows.clone(),
            head_dim,
            cols: (wire.engram_max_ngram - 1) * wire.engram_heads,
            stride,
            hist: Vec::new(),
            ring,
        }))
    }

    /// Fill the pinned rows for the block at `pos0..pos0+tokens.len()`.
    ///
    /// The forward hands over the exact block it is about to run -- including
    /// draft tokens a verify batch may later reject; the next block's write
    /// overwrites those slots before any read, the engine's hist memcpy rule
    /// (`core_v41_forward.c:377`).  `dst[k]` is engram layer k's pinned
    /// buffer, `[n][cols][stride]`: per row the e4m3 plane then the ue8m0
    /// tail, the on-disk row form.
    pub fn prepare(&mut self, pos0: u32, tokens: &[i32], dst: &[*mut u8]) -> std::result::Result<(), String> {
        let start = pos0 as usize;
        let end = start + tokens.len();
        if self.hist.len() < end {
            self.hist.resize(end, 0);
        }
        self.hist[start..end].copy_from_slice(tokens);
        if self.ring.is_some() {
            for (k, &base) in dst.iter().enumerate() {
                self.prepare_round(k as u32, start, end, base)?;
            }
            return Ok(());
        }
        for (k, &base) in dst.iter().enumerate() {
            let ei = k as u32;
            for (i, p) in (start..end).enumerate() {
                let rows = self.hash.rows(&self.hist, p as i64, ei);
                for (c, &row_id) in rows.iter().enumerate() {
                    if row_id < 0 || row_id as u64 >= self.rows[k] {
                        return Err(format!(
                            "engram row {row_id} out of range (layer {k}, rows {})",
                            self.rows[k]
                        ));
                    }
                    let row = self.shards[k]
                        .row(self.weight_off[k], self.scale_off[k], row_id as u64, self.head_dim)
                        .map_err(|e| e.to_string())?;
                    let slot = (i * self.cols as usize + c) * self.stride as usize;
                    unsafe {
                        let dst = base.add(slot);
                        std::ptr::copy_nonoverlapping(row.weights.as_ptr(), dst, row.weights.len());
                        std::ptr::copy_nonoverlapping(
                            row.scale.as_ptr(),
                            dst.add(row.weights.len()),
                            row.scale.len(),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// One engram layer's round on the ring, the engine's own shape
    /// (`core_v41_engram.c:238-254`): hash the block's rows on this thread,
    /// fill one request per plane per row (weights then scales), one submit
    /// for the whole round, one wait.  The row-range refusal is the engine's
    /// err=1 (`core_v41_engram.c:247`).
    fn prepare_round(
        &mut self,
        ei: u32,
        start: usize,
        end: usize,
        base: *mut u8,
    ) -> std::result::Result<(), String> {
        let ring = self.ring.as_mut().expect("ring round without a ring");
        let hd = self.head_dim as u64;
        let nsc = hd / 32;
        let k = ei as usize;
        let direct = self.shards[k].is_direct();
        let fd = self.shards[k].raw_fd();
        let reqs = ring.reqs_mut();
        reqs.clear();
        reqs.reserve((end - start) * self.cols as usize * 2);
        for (i, p) in (start..end).enumerate() {
            let rows = self.hash.rows(&self.hist, p as i64, ei);
            for (c, &row_id) in rows.iter().enumerate() {
                if row_id < 0 || row_id as u64 >= self.rows[k] {
                    return Err(format!(
                        "engram row {row_id} out of range (layer {k}, rows {})",
                        self.rows[k]
                    ));
                }
                let r = row_id as u64;
                let slot = (i * self.cols as usize + c) * self.stride as usize;
                let wdst = unsafe { base.add(slot) };
                reqs.push(Req {
                    fd,
                    direct,
                    len: hd as u32,
                    off: self.weight_off[k] + r * hd,
                    dst: wdst,
                });
                reqs.push(Req {
                    fd,
                    direct,
                    len: nsc as u32,
                    off: self.scale_off[k] + r * nsc,
                    dst: unsafe { wdst.add(hd as usize) },
                });
            }
        }
        if !ring.submit() || !ring.wait() {
            return Err("engram io_uring round failed".into());
        }
        Ok(())
    }

    /// The pinned row stride the native buffer uses (for the native buffer
    /// sizing test / diagnostics).
    pub fn stride(&self) -> u32 {
        self.stride
    }
}

struct FeedTramp<'a> {
    feed: &'a mut V41Feed,
    n_layers: usize,
}

unsafe extern "C" fn rows_tramp(
    pos0: u32,
    tokens: *const i32,
    n: i32,
    dst: *const *mut u8,
    ud: *mut c_void,
) -> i32 {
    if ud.is_null() || dst.is_null() || (n > 0 && tokens.is_null()) || n < 0 {
        return 1;
    }
    let t = &mut *(ud as *mut FeedTramp<'_>);
    let tokens = std::slice::from_raw_parts(tokens, n as usize);
    let dst = std::slice::from_raw_parts(dst, t.n_layers);
    match catch_unwind(AssertUnwindSafe(|| t.feed.prepare(pos0, tokens, dst))) {
        Ok(Ok(())) => 0,
        // A provider failure (or panic) refuses the block; the forward fails
        // loud instead of running on stale rows.
        _ => 1,
    }
}

struct EmitTramp<'a> {
    emit: &'a mut dyn FnMut(i32) -> bool,
}

unsafe extern "C" fn emit_tramp(token: i32, ud: *mut c_void) -> i32 {
    if ud.is_null() {
        return 1;
    }
    let t = &mut *(ud as *mut EmitTramp<'_>);
    match catch_unwind(AssertUnwindSafe(|| (t.emit)(token))) {
        Ok(true) => 0,
        // false = stop, panic = stop: never keep generating past a dead sink.
        _ => 1,
    }
}

struct ProgressTramp<'a> {
    progress: &'a mut dyn FnMut(&str, i32, i32) -> bool,
}

unsafe extern "C" fn progress_tramp(ud: *mut c_void, event: *const c_char, current: i32, total: i32) -> i32 {
    if ud.is_null() {
        return 1;
    }
    let t = &mut *(ud as *mut ProgressTramp<'_>);
    let event = if event.is_null() {
        ""
    } else {
        std::ffi::CStr::from_ptr(event).to_str().unwrap_or("")
    };
    match catch_unwind(AssertUnwindSafe(|| (t.progress)(event, current, total))) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(_) => 1,
    }
}

fn asset_error(e: impl std::fmt::Display) -> Error {
    Error {
        code: 1,
        message: e.to_string(),
    }
}

/// Mount the ②/③ sidecar for one run (the engine's per-state
/// `v41_state_plugins` -> `v41_amp_load`, `core_v41_state.c:77-87` /
/// `core_v41_amp.c:227-270`): the process-wide rb table unloads first
/// (`core_v41_amp.c:114`), then every present layer's rb delta and gr
/// gain table land on the device (rb before gr, the engine's order at
/// `:228-230`).  The files are read and merged by the P2/P3 Rust reader
/// (`zchain.rs`); the device writes are the P4-7 stores.  No directories =
/// the engine's naked base: only the rb unload (`core_v41_state.c:78`).
///
/// A directory carrying `amp_Lnn.bin` is refused by name: the low-rank
/// apply is not ported (the recorded §6.14 limit; no Spark directory
/// carries one).  The refusal fires before any device write, so a failed
/// mount never leaves a half-mounted process (the engine mounts rb/gr
/// first and fails at the amp read; the request outcome is the same).
fn v41_mount_zchain(
    model: &Model,
    path: &Path,
    zchain: Option<&str>,
    posttrain: Option<&str>,
) -> Result<()> {
    if zchain.is_none() && posttrain.is_none() {
        unsafe { ds4_bridge_v41_zchain_rb_clear() };
        return Ok(());
    }
    let g = GgufFile::open(path).map_err(asset_error)?;
    let shape = identify_file(&g).map_err(asset_error)?.shape;
    let geom = ZchainGeom {
        n_layer: shape.n_layer,
        n_expert: shape.n_expert,
        n_embd: shape.n_embd,
    };
    let merged = zchain_load_for_run(zchain, posttrain, geom)?;
    // Unreachable with a directory given (load answers None only for two
    // Nones); kept total rather than panicking.
    let Some(merged) = merged else {
        return Ok(());
    };
    unsafe { ds4_bridge_v41_zchain_rb_clear() };
    for (il, delta) in merged.rb.iter().enumerate() {
        let Some(delta) = delta else { continue };
        let ok = unsafe {
            ds4_bridge_v41_zchain_rb_set(model.raw_ptr(), il as u32, delta.as_ptr(), geom.n_expert)
        };
        if ok == 0 {
            return Err(asset_error(format!("V4.1 zchain: rb L{il:02} upload refused")));
        }
    }
    for (il, gr) in merged.gr.iter().enumerate() {
        let Some(gr) = gr else { continue };
        let ok = unsafe {
            ds4_bridge_v41_zchain_gr_set(
                model.raw_ptr(),
                il as u32,
                gr.factor.as_ptr(),
                geom.n_expert,
                geom.n_embd,
            )
        };
        if ok == 0 {
            return Err(asset_error(format!("V4.1 zchain: gr L{il:02} upload refused")));
        }
    }
    eprintln!(
        "ds4: [v41] zchain mount: gr={} rb={} amp=0 from {}{}",
        merged.n_gr_layers(),
        merged.n_rb_layers(),
        zchain.unwrap_or("(none)"),
        if posttrain.is_some() { " + posttrain" } else { "" },
    );
    Ok(())
}

/// The load-and-refuse half of the mount, split out so it is unit-testable
/// without a model: the P2/P3 reader/merger plus the port's recorded limit
/// — a directory carrying `amp_Lnn.bin` is refused by name because the
/// low-rank apply is not ported (§6.14; no Spark directory carries one).
/// No device state is touched here, so a refusal never leaves a partial
/// mount behind.
fn zchain_load_for_run(
    zchain: Option<&str>,
    posttrain: Option<&str>,
    geom: ZchainGeom,
) -> Result<Option<V41Zchain>> {
    let merged = V41Zchain::load(zchain.map(Path::new), posttrain.map(Path::new), geom, 1.0)
        .map_err(|e| asset_error(format!("V4.1 zchain refused: {}", e.token())))?;
    let Some(merged) = merged else {
        return Ok(None);
    };
    if merged.n_amp_layers() > 0 {
        return Err(asset_error(
            "V4.1 zchain carries amp_Lnn.bin; the low-rank amplifier apply is not ported",
        ));
    }
    Ok(Some(merged))
}

impl Model {
    /// Run the engine's one-shot V4.1 generate
    /// (`ds4_engine_v41_generate_argmax`, `core_v41_api.c:426-470` + the
    /// round drive `:175-425`) with the host feed.
    ///
    /// `emit` is called per produced token in order; returning false stops
    /// the run (the CLI's EOS convention -- EOS itself also stops inside the
    /// native).  `progress` is called once per prefill chunk; returning false
    /// aborts the prefill (the server's client-gone check).  Both run on the
    /// calling thread.  `path` is the model file the feed re-reads its
    /// metadata and shard paths from (the model handle itself keeps no path).
    pub fn v41_generate(
        &self,
        path: &Path,
        prompt: &[i32],
        n_predict: i32,
        opts: &V41RunOptions<'_>,
        emit: &mut dyn FnMut(i32) -> bool,
        progress: &mut dyn FnMut(&str, i32, i32) -> bool,
    ) -> Result<()> {
        if prompt.is_empty() {
            return Err(asset_error("V4.1 prompt is empty"));
        }
        if n_predict < 0 {
            return Err(asset_error("V4.1 n_predict is negative"));
        }
        // The switches are process globals in the native (the engine's CLI
        // sets them before the run); set them here so every entry point lands
        // on the same knob.  None leaves the native default untouched.
        unsafe {
            if let Some(mode) = opts.dspark {
                ds4_bridge_v41_set_dspark(mode);
            }
            if let Some(on) = opts.graph {
                ds4_bridge_v41_set_graph(i32::from(on));
            }
            ds4_bridge_v41_set_emit_trace(i32::from(opts.emit_trace));
            ds4_bridge_v41_set_prof(i32::from(opts.prof));
            // The sampling face is per request (the engine's server sets it
            // right before the generate, server_generate_v41.c:409); None
            // resets it, so a run never inherits the previous request's face.
            match opts.sampling {
                Some(s) => {
                    let face = ds4_sys::ds4_bridge_v41_sampling {
                        temperature: s.temperature,
                        top_p: s.top_p,
                        min_p: s.min_p,
                        top_k: s.top_k,
                        seed: s.seed,
                        freq_penalty: s.freq_penalty,
                        presence_penalty: s.presence_penalty,
                        dry_multiplier: s.dry_multiplier,
                        dry_base: s.dry_base,
                        dry_allowed_length: s.dry_allowed_length,
                    };
                    ds4_bridge_v41_set_sampling(&face);
                }
                None => ds4_bridge_v41_set_sampling(std::ptr::null()),
            }
        }
        // The ②/③ sidecar mount lands before the native call, exactly like
        // the engine's per-state mount inside the generate (the device
        // tables are process-level; the forward reads them).
        v41_mount_zchain(self, path, opts.zchain, opts.posttrain)?;
        let mut feed = if opts.no_engram {
            None
        } else {
            V41Feed::open(path, opts.engram_dir)?
        };
        // The tramp borrows the feed for the duration of the call only; the
        // native never retains the pointer past it (ds4_bridge.h).
        let mut feed_tramp = feed.as_mut().map(|f| FeedTramp {
            n_layers: f.shards.len(),
            feed: f,
        });
        let (rows_fn, rows_ud): (ds4_sys::ds4_bridge_v41_rows_fn, *mut c_void) =
            match feed_tramp.as_mut() {
                Some(t) => (Some(rows_tramp), t as *mut FeedTramp<'_> as *mut c_void),
                None => (None, std::ptr::null_mut()),
            };
        let mut emit_tramp_state = EmitTramp { emit };
        let mut progress_tramp_state = ProgressTramp { progress };
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_bridge_v41_generate(
                self.raw_ptr(),
                prompt.as_ptr(),
                prompt.len() as i32,
                n_predict,
                i32::from(opts.no_engram),
                opts.verify_k,
                rows_fn,
                rows_ud,
                Some(emit_tramp),
                &mut emit_tramp_state as *mut EmitTramp<'_> as *mut c_void,
                Some(progress_tramp),
                &mut progress_tramp_state as *mut ProgressTramp<'_> as *mut c_void,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if rc != 0 {
            return Err(crate::fail(rc, &err));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic sidecar files (the layouts the readers check,
    /// `core_v41_amp.c:44-84` / `:193-220`).
    fn put_i32(buf: &mut Vec<u8>, v: i32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn gr_f32(n_expert: i32, d: i32) -> Vec<u8> {
        let mut b = Vec::new();
        put_i32(&mut b, n_expert);
        put_i32(&mut b, d);
        put_i32(&mut b, 1);
        for i in 0..(n_expert * d) {
            b.extend_from_slice(&(1.0f32 + i as f32 * 0.001).to_le_bytes());
        }
        b
    }

    fn amp_f32(d: i32, k: i32) -> Vec<u8> {
        let mut b = Vec::new();
        put_i32(&mut b, d);
        put_i32(&mut b, k);
        put_i32(&mut b, 1);
        for i in 0..(2 * k * d) {
            b.extend_from_slice(&(0.1f32 * i as f32).to_le_bytes());
        }
        b
    }

    fn tmpdir(name: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ds4-v41-mount-{n}-{name}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    const GEOM: ZchainGeom = ZchainGeom {
        n_layer: 2,
        n_expert: 4,
        n_embd: 2,
    };

    #[test]
    fn no_dirs_answers_none() {
        assert!(zchain_load_for_run(None, None, GEOM).unwrap().is_none());
    }

    #[test]
    fn gr_only_dir_loads_and_counts() {
        let dir = tmpdir("gr-only");
        std::fs::write(dir.join("gr_L00.bin"), gr_f32(4, 2)).unwrap();
        let merged = zchain_load_for_run(Some(dir.to_str().unwrap()), None, GEOM)
            .unwrap()
            .unwrap();
        assert_eq!(merged.n_gr_layers(), 1);
        assert_eq!(merged.n_rb_layers(), 0);
        assert_eq!(merged.n_amp_layers(), 0);
    }

    #[test]
    fn amp_layer_is_refused_by_name() {
        let dir = tmpdir("amp");
        std::fs::write(dir.join("gr_L00.bin"), gr_f32(4, 2)).unwrap();
        std::fs::write(dir.join("amp_L00.bin"), amp_f32(2, 1)).unwrap();
        let err = zchain_load_for_run(Some(dir.to_str().unwrap()), None, GEOM).unwrap_err();
        assert!(
            err.message.contains("amp_Lnn.bin"),
            "refusal must name amp_Lnn.bin, got: {}",
            err.message
        );
    }

    #[test]
    fn empty_dir_is_refused() {
        let dir = tmpdir("empty");
        let err = zchain_load_for_run(Some(dir.to_str().unwrap()), None, GEOM).unwrap_err();
        assert!(
            err.message.contains("zchain-empty"),
            "expected the engine's empty-plugin refusal, got: {}",
            err.message
        );
    }
}
