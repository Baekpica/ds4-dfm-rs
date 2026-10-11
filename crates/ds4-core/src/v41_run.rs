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
use std::sync::{mpsc, Arc, Condvar, Mutex};

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

/// The per-run switches (the engine's CLI globals, `cli_diag.c:58`).  No
/// Clone/Debug derive: the greedy hook holds a `&mut dyn FnMut`, which neither
/// implements.
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
    /// The tool-syntax greedy hook (the engine's ds4_greedy_fn, c6e2706): the
    /// serve's closure; the native asks once per position.  None = no hook.
    pub greedy: Option<&'a mut dyn FnMut(*const i32, u32) -> bool>,
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
            greedy: None,
            zchain: None,
            posttrain: None,
        }
    }
}

/// The engram rows provider: the hash tables and one open shard per engram
/// layer, plus the token history the n-gram window reads.
///
/// P2 splits the engine's read protocol out of `prepare` (the engine's
/// `v41_engram_prefetch`, `core_v41_engram.c:311-341`): `prepare` retires the
/// previous block, records the history and submits round 0 only (or wakes the
/// fetch worker for an oversize block); the native forward collects each
/// layer's round through the wait hook at its engram layer (the engine's
/// `v41_ejob_wait`, `:347-379`) and only then uploads the rows.  The
/// synchronous protocol (no ring, `DS41_FEED_SYNC`, or a worker that failed
/// to start) stays exactly the old path: `prepare` fills everything before
/// returning and no wait hook is installed.
pub struct V41Feed {
    /// The ring-path state, shared with the fetch worker; the mode decides
    /// which thread touches it (`core_v41_engram.c:104-125`).
    core: Arc<Mutex<FeedCore>>,
    /// The fetch worker for oversize rounds (the engine's `v41_ejob_io_thread`).
    worker: Option<Worker>,
    /// The worker's control slots (the engine's `io_mu`/`io_go`/`io_done`/`err`).
    ctl: Arc<(Mutex<Ctl>, Condvar)>,
    /// The async protocol is on (a ring exists, no kill switch, worker up).
    async_on: bool,
    has_ring: bool,
    ring_cap: u32,
    cols: u32,
    stride: u32,
    n_layers: usize,
    /// This block's round mode and per-layer bookkeeping (the engine's
    /// `io_mode`/`started`/`joined`).
    io_mode: bool,
    prev_worker: bool,
    started: Vec<bool>,
    joined: Vec<bool>,
    block_start: usize,
    block_end: usize,
    bases: Vec<usize>,
    /// Sticky failure: once a round failed every later call fails loud (the
    /// engine's `J->err`; the run aborts on the first refusal either way).
    err: Option<String>,
    /// The controlled-delay gate hooks (the P2 review's test): delay each
    /// layer's collect/publication by N us (a slow disk), or refuse every
    /// wait (the native route must then fail loud, never consume stale rows).
    /// Read once at open, so a run is one configuration.
    wait_delay_us: Option<u64>,
    wait_withhold: bool,
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
        let Some(hash) =
            EngramHash::load(&g, &inv, &wire, identified.shape.n_vocab).map_err(asset_error)?
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
        // P2 protocol (the engine's prefetch/wait split, core_v41_engram.c:311-379):
        // a ring and no DS41_FEED_SYNC installs the async protocol -- prepare
        // submits round 0 only, the native waits per layer; anything else
        // keeps the synchronous prepare, bit for bit the old path.
        let has_ring = ring.is_some();
        let ring_cap = ring.as_ref().map_or(0, |r| r.cap());
        let mut async_on = has_ring && std::env::var_os("DS41_FEED_SYNC").is_none();
        if has_ring && !async_on {
            eprintln!("ds4: [v41] engram rows: synchronous protocol (DS41_FEED_SYNC)");
        }
        let n_layers = shards.len();
        let cols = (wire.engram_max_ngram - 1) * wire.engram_heads;
        let core = Arc::new(Mutex::new(FeedCore {
            ring,
            hash,
            shards,
            weight_off: wire.engram_weight_off.clone(),
            scale_off: wire.engram_scale_off.clone(),
            rows: wire.engram_rows.clone(),
            head_dim,
            cols,
            stride,
            hist: Vec::new(),
        }));
        let ctl = Arc::new((
            Mutex::new(Ctl {
                go: false,
                done: 0,
                err: None,
            }),
            Condvar::new(),
        ));
        // The fetch worker (the engine's v41_ejob_io_thread): an oversize
        // round runs here so the calling thread never drains tens of
        // thousands of reads at the engram layer.  A thread that refuses to
        // start falls back to the synchronous path (the engine's own
        // fallback, :327-332).
        let worker = if async_on {
            let (tx, rx) = mpsc::channel();
            let spawned = std::thread::Builder::new()
                .name("ds41-engram-fetch".into())
                .spawn({
                    let core = Arc::clone(&core);
                    let ctl = Arc::clone(&ctl);
                    move || worker_loop(core, ctl, rx)
                });
            match spawned {
                Ok(join) => Some(Worker {
                    tx,
                    join: Some(join),
                }),
                Err(_) => {
                    eprintln!(
                        "ds4: [v41] engram fetch thread refused to start; main-thread rounds"
                    );
                    None
                }
            }
        } else {
            None
        };
        if worker.is_none() {
            async_on = false;
        }
        let wait_delay_us = std::env::var("DS41_FEED_WAIT_DELAY_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&us| us > 0);
        let wait_withhold = std::env::var_os("DS41_FEED_WAIT_WITHHOLD").is_some();
        if wait_delay_us.is_some() || wait_withhold {
            eprintln!(
                "ds4: [v41] engram wait test hooks: delay {:?} us, withhold {}",
                wait_delay_us, wait_withhold
            );
        }
        Ok(Some(Self {
            core,
            worker,
            ctl,
            async_on,
            has_ring,
            ring_cap,
            cols,
            stride,
            n_layers,
            io_mode: false,
            prev_worker: false,
            started: vec![false; n_layers],
            joined: vec![false; n_layers],
            block_start: 0,
            block_end: 0,
            bases: Vec::new(),
            err: None,
            wait_delay_us,
            wait_withhold,
        }))
    }

    /// The engine's `v41_engram_prefetch` (`core_v41_engram.c:311-341`):
    /// retire the previous block's rounds, record the history, then submit
    /// round 0 only (small block) or wake the fetch worker (oversize round);
    /// the synchronous protocol fills every layer here, exactly the old path.
    ///
    /// The forward hands over the exact block it is about to run -- including
    /// draft tokens a verify batch may later reject; the next block's write
    /// overwrites those slots before any read, the engine's hist memcpy rule
    /// (`core_v41_forward.c:377`).  `dst[k]` is engram layer k's pinned
    /// buffer, `[n][cols][stride]`: per row the e4m3 plane then the ue8m0
    /// tail, the on-disk row form.
    pub fn prepare(
        &mut self,
        pos0: u32,
        tokens: &[i32],
        dst: &[*mut u8],
    ) -> std::result::Result<(), String> {
        let r = self.prepare_inner(pos0, tokens, dst);
        if let Err(e) = &r {
            self.err = Some(e.clone());
        }
        r
    }

    fn prepare_inner(
        &mut self,
        pos0: u32,
        tokens: &[i32],
        dst: &[*mut u8],
    ) -> std::result::Result<(), String> {
        if let Some(e) = &self.err {
            return Err(e.clone());
        }
        if dst.len() < self.n_layers {
            return Err(format!(
                "engram feed got {} layer buffers, the model has {}",
                dst.len(),
                self.n_layers
            ));
        }
        self.retire()?;
        let start = pos0 as usize;
        let end = start + tokens.len();
        self.block_start = start;
        self.block_end = end;
        self.bases.clear();
        self.bases.extend(dst.iter().map(|&p| p as usize));
        // A round that does not fit the queue goes to the worker
        // (`core_v41_engram.c:325`); the ring's own capacity decides, not a
        // constant (the kernel may return a different depth).
        self.io_mode =
            self.async_on && (tokens.len() as u64 * self.cols as u64 * 2u64) > self.ring_cap as u64;
        {
            let mut c = self.core.lock().unwrap();
            c.splice_hist(start, tokens);
            if !self.async_on {
                if self.has_ring {
                    for k in 0..self.n_layers {
                        c.fill_round(k as u32, start, end, dst[k])?;
                        c.collect()?;
                    }
                } else {
                    for k in 0..self.n_layers {
                        c.serial_round(k as u32, start, end, dst[k])?;
                    }
                }
            } else if !self.io_mode {
                c.fill_round(0, start, end, dst[0])?;
            }
        }
        if self.io_mode {
            {
                let (m, _) = &*self.ctl;
                let mut g = m.lock().unwrap();
                g.go = true;
                g.done = 0;
                g.err = None;
            }
            let cmd = FeedCmd::Rounds {
                start,
                end,
                bases: self.bases.clone(),
                n_eng: self.n_layers,
            };
            let sent = self
                .worker
                .as_ref()
                .map_or(false, |w| w.tx.send(cmd).is_ok());
            if !sent {
                let (m, _) = &*self.ctl;
                m.lock().unwrap().go = false;
                return Err("engram fetch worker is gone".into());
            }
        }
        for k in 0..self.n_layers {
            if !self.async_on {
                self.started[k] = true;
                self.joined[k] = true;
            } else if self.io_mode {
                self.started[k] = true;
                self.joined[k] = false;
            } else {
                self.started[k] = k == 0;
                self.joined[k] = false;
            }
        }
        self.prev_worker = self.io_mode;
        Ok(())
    }

    /// The engine's `v41_ejob_wait` (`core_v41_engram.c:347-379`): collect
    /// layer `k`'s round.  Idempotent; a round never started is submitted
    /// here after every earlier started-unjoined round is collected
    /// (`:360-363`); collecting round k chains round k+1's submit (`:376-377`)
    /// so the layers between them cover its reads.  With the worker the call
    /// waits the layer's completion bit instead (`:350-358`).
    pub fn wait_layer(&mut self, k: u32) -> std::result::Result<(), String> {
        let r = self.wait_layer_inner(k);
        if let Err(e) = &r {
            self.err = Some(e.clone());
        }
        r
    }

    fn wait_layer_inner(&mut self, k: u32) -> std::result::Result<(), String> {
        if let Some(e) = &self.err {
            return Err(e.clone());
        }
        let k = k as usize;
        if k >= self.n_layers {
            return Err(format!(
                "engram wait layer {k} out of range ({} layers)",
                self.n_layers
            ));
        }
        if !self.async_on {
            return Ok(()); // the synchronous protocol collected every round in prepare
        }
        if self.joined[k] {
            return Ok(());
        }
        // The controlled-delay gate hooks (P2 review test): the delay mimics
        // a slow disk (publication lands late -> the graph's spin kernel
        // waits), the withhold makes every collect fail so the native must
        // refuse rather than consume stale rows.
        if let Some(us) = self.wait_delay_us {
            std::thread::sleep(std::time::Duration::from_micros(us));
        }
        if self.wait_withhold {
            return Err("engram wait withheld (DS41_FEED_WAIT_WITHHOLD)".into());
        }
        if self.io_mode {
            let (m, cv) = &*self.ctl;
            let mut g = m.lock().unwrap();
            while (g.done & (1u32 << k)) == 0 && g.err.is_none() {
                g = cv.wait(g).unwrap();
            }
            let err = g.err.clone();
            drop(g);
            self.joined[k] = true;
            if let Some(e) = err {
                return Err(e);
            }
            return Ok(());
        }
        let mut c = self.core.lock().unwrap();
        if !self.started[k] {
            for j in 0..k {
                if self.started[j] && !self.joined[j] {
                    c.collect()?;
                    self.joined[j] = true;
                }
            }
            c.fill_round(
                k as u32,
                self.block_start,
                self.block_end,
                self.bases[k] as *mut u8,
            )?;
            self.started[k] = true;
        }
        if !self.joined[k] {
            c.collect()?;
            self.joined[k] = true;
            if k + 1 < self.n_layers && !self.started[k + 1] {
                c.fill_round(
                    (k + 1) as u32,
                    self.block_start,
                    self.block_end,
                    self.bases[k + 1] as *mut u8,
                )?;
                self.started[k + 1] = true;
            }
        }
        Ok(())
    }

    /// The engine's prefetch retire (`core_v41_engram.c:313-319`): the
    /// previous block's rounds must be done before the history and the
    /// request vector are touched again -- a completion indexes the resident
    /// request vector and copies into its baked destination (`uring.rs`
    /// wait), so a stale round must never outlive its block.  Covers the
    /// paths where a consumer never ran: a CED block that skips engram
    /// layers, a failed capture whose step re-runs direct, teardown.
    fn retire(&mut self) -> std::result::Result<(), String> {
        if self.prev_worker {
            let (m, cv) = &*self.ctl;
            let mut g = m.lock().unwrap();
            while g.go {
                if self.worker_is_dead() {
                    return Err("engram fetch worker died with rounds in flight".into());
                }
                g = cv.wait(g).unwrap();
            }
            let err = g.err.clone();
            for k in 0..self.n_layers {
                self.joined[k] = true;
            }
            if let Some(e) = err {
                return Err(e);
            }
            return Ok(());
        }
        if !self.async_on || !self.has_ring {
            return Ok(());
        }
        let mut c = self.core.lock().unwrap();
        for k in 0..self.n_layers {
            if self.started[k] && !self.joined[k] {
                c.collect()?;
                self.joined[k] = true;
            }
        }
        Ok(())
    }

    fn worker_is_dead(&self) -> bool {
        self.worker
            .as_ref()
            .map_or(true, |w| w.join.as_ref().map_or(true, |h| h.is_finished()))
    }

    /// Whether the native side should install the wait hook (the async
    /// protocol); false keeps the synchronous `prepare` (no ring, the
    /// `DS41_FEED_SYNC` kill switch, or a worker that failed to start).
    pub fn async_protocol(&self) -> bool {
        self.async_on
    }

    /// The pinned row stride the native buffer uses (for the native buffer
    /// sizing test / diagnostics).
    pub fn stride(&self) -> u32 {
        self.stride
    }
}

/// The ring-path state (the engine's `v41_ejob`, `core_v41_engram.c:104-125`):
/// the hash tables, the open shards, the token history, the io_uring channel
/// and the resident request vector.  The calling thread runs the small
/// (decode) rounds and the fetch worker runs the oversize (prefill) rounds
/// (`core_v41_engram.c:128-150`); the mutex is the exclusivity between them,
/// so both may hold the type and never the ring at once.
struct FeedCore {
    ring: Option<crate::uring::Ring>,
    hash: EngramHash,
    shards: Vec<EngramShard>,
    weight_off: Vec<u64>,
    scale_off: Vec<u64>,
    rows: Vec<u64>,
    head_dim: u32,
    cols: u32,
    stride: u32,
    hist: Vec<i32>,
}

impl FeedCore {
    /// The engine's history write (`core_v41_forward.c:377`): the block's
    /// tokens land at their absolute positions before any hash reads them.
    fn splice_hist(&mut self, start: usize, tokens: &[i32]) {
        let end = start + tokens.len();
        if self.hist.len() < end {
            self.hist.resize(end, 0);
        }
        self.hist[start..end].copy_from_slice(tokens);
    }

    /// One layer's round on the ring, the engine's own shape
    /// (`core_v41_engram.c:238-254`): hash the block's rows on this thread,
    /// fill one request per plane per row (weights then scales), one submit
    /// for the whole round.  The row-range refusal is the engine's err=1
    /// (`core_v41_engram.c:247`).
    fn fill_round(
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
        if !ring.submit() {
            return Err("engram io_uring submit failed".into());
        }
        Ok(())
    }

    /// `v41_ering_wait` (`core_v41_ering.c:122-144`): drain the round's
    /// completions, re-pushing the remainder; the engine's failure shape is
    /// `v41_ejob_wait`'s err=2 (`core_v41_engram.c:367-368`).
    fn collect(&mut self) -> std::result::Result<(), String> {
        let ring = self.ring.as_mut().expect("ring collect without a ring");
        if !ring.wait() {
            return Err("engram io_uring round failed".into());
        }
        Ok(())
    }

    /// The no-ring fallback (the engine's thread-pool-down path,
    /// `core_v41_engram.c:301-308`): serial preads for one layer.
    fn serial_round(
        &mut self,
        ei: u32,
        start: usize,
        end: usize,
        base: *mut u8,
    ) -> std::result::Result<(), String> {
        let k = ei as usize;
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
                    .row(
                        self.weight_off[k],
                        self.scale_off[k],
                        row_id as u64,
                        self.head_dim,
                    )
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
        Ok(())
    }
}

/// The fetch worker's control slots (the engine's `io_go`/`io_done`/`err`
/// under `io_mu`, `core_v41_engram.c:121-125`).
struct Ctl {
    go: bool,
    done: u32,
    err: Option<String>,
}

/// One oversize block for the worker (the engine's per-layer round loop,
/// `core_v41_engram.c:128-150`).
enum FeedCmd {
    Rounds {
        start: usize,
        end: usize,
        bases: Vec<usize>,
        n_eng: usize,
    },
    Stop,
}

struct Worker {
    tx: mpsc::Sender<FeedCmd>,
    join: Option<std::thread::JoinHandle<()>>,
}

/// The engine's `v41_ejob_io_thread` (`core_v41_engram.c:128-150`): per
/// layer, hash + submit (`fill_round`) then collect (`collect`), publish the
/// layer's completion bit; on failure publish every bit with the error kept
/// so the waiters stop (`:136-141`).  A panic becomes the same error shape.
fn worker_loop(
    core: Arc<Mutex<FeedCore>>,
    ctl: Arc<(Mutex<Ctl>, Condvar)>,
    rx: mpsc::Receiver<FeedCmd>,
) {
    loop {
        let cmd = match rx.recv() {
            Ok(c) => c,
            Err(_) => break,
        };
        let (start, end, bases, n_eng) = match cmd {
            FeedCmd::Rounds {
                start,
                end,
                bases,
                n_eng,
            } => (start, end, bases, n_eng),
            FeedCmd::Stop => break,
        };
        let mut failed = false;
        for ei in 0..n_eng {
            let r = catch_unwind(AssertUnwindSafe(|| {
                let mut c = core.lock().unwrap();
                c.fill_round(ei as u32, start, end, bases[ei] as *mut u8)
                    .and_then(|_| c.collect())
            }))
            .unwrap_or_else(|_| Err("engram fetch worker panicked".into()));
            let (m, cv) = &*ctl;
            let mut g = m.lock().unwrap();
            match r {
                Ok(()) => {
                    g.done |= 1 << ei;
                }
                Err(e) => {
                    if g.err.is_none() {
                        g.err = Some(e);
                    }
                    g.done = (1u32 << n_eng) - 1;
                    failed = true;
                }
            }
            cv.notify_all();
            drop(g);
            if failed {
                break;
            }
        }
        let (m, cv) = &*ctl;
        let mut g = m.lock().unwrap();
        g.go = false;
        cv.notify_all();
    }
}

impl Drop for V41Feed {
    fn drop(&mut self) {
        if self.worker.is_none() {
            return;
        }
        // Finish any in-flight block before the thread goes: the pinned
        // buffers it writes die with the native feed right after this.
        {
            let (m, cv) = &*self.ctl;
            let mut g = m.lock().unwrap();
            while g.go {
                if self.worker_is_dead() {
                    break;
                }
                g = cv.wait(g).unwrap();
            }
        }
        if let Some(w) = &mut self.worker {
            let _ = w.tx.send(FeedCmd::Stop);
            if let Some(h) = w.join.take() {
                let _ = h.join();
            }
        }
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

/// The native wait hook (P2): the forward collects layer k's round here at
/// its engram layer (the engine's `v41_ejob_wait` call site,
/// `core_v41_engram.c:441`); a nonzero return refuses the block before the
/// rows are uploaded.
unsafe extern "C" fn wait_tramp(k: i32, ud: *mut c_void) -> i32 {
    if ud.is_null() || k < 0 {
        return 1;
    }
    let t = &mut *(ud as *mut FeedTramp<'_>);
    match catch_unwind(AssertUnwindSafe(|| t.feed.wait_layer(k as u32))) {
        Ok(Ok(())) => 0,
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

unsafe extern "C" fn progress_tramp(
    ud: *mut c_void,
    event: *const c_char,
    current: i32,
    total: i32,
) -> i32 {
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

/// The tool-syntax greedy hook (the engine's v41_syntax_greedy ud, c6e2706):
/// the serve's closure; the native asks once per position through this tramp
/// (the emitted text + the ahead tokens; nonzero = that position takes the raw
/// argmax).  The closure lives in the caller's frame for the call's duration.
struct GreedyTramp<'a> {
    greedy: &'a mut dyn FnMut(*const i32, u32) -> bool,
}

unsafe extern "C" fn greedy_tramp(ud: *mut c_void, ahead: *const i32, n: u32) -> i32 {
    if ud.is_null() {
        return 0;
    }
    let t = &mut *(ud as *mut GreedyTramp<'_>);
    match catch_unwind(AssertUnwindSafe(|| (t.greedy)(ahead, n))) {
        Ok(true) => 1,
        _ => 0,
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
            return Err(asset_error(format!(
                "V4.1 zchain: rb L{il:02} upload refused"
            )));
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
            return Err(asset_error(format!(
                "V4.1 zchain: gr L{il:02} upload refused"
            )));
        }
    }
    eprintln!(
        "ds4: [v41] zchain mount: gr={} rb={} amp=0 from {}{}",
        merged.n_gr_layers(),
        merged.n_rb_layers(),
        zchain.unwrap_or("(none)"),
        if posttrain.is_some() {
            " + posttrain"
        } else {
            ""
        },
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
        opts: &mut V41RunOptions<'_>,
        emit: &mut dyn FnMut(i32) -> bool,
        progress: &mut dyn FnMut(&str, i32, i32) -> bool,
    ) -> Result<()> {
        if prompt.is_empty() {
            return Err(asset_error("V4.1 prompt is empty"));
        }
        if n_predict < 0 {
            return Err(asset_error("V4.1 n_predict is negative"));
        }
        // The tool-syntax greedy hook (the engine's v41_syntax_greedy,
        // c6e2706): the serve's closure; the tramp borrows it for the call's
        // duration only.  None leaves the native asking nobody (bare argmax).
        let mut greedy_ctx = opts
            .greedy
            .as_mut()
            .map(|g| GreedyTramp { greedy: &mut **g });
        let greedy_fn = greedy_ctx
            .as_ref()
            .map(|_| greedy_tramp as unsafe extern "C" fn(*mut c_void, *const i32, u32) -> i32);
        let greedy_ud = greedy_ctx.as_mut().map_or(std::ptr::null_mut(), |t| {
            t as *mut GreedyTramp<'_> as *mut c_void
        });
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
                        greedy_fn,
                        greedy_ud,
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
            n_layers: f.n_layers,
            feed: f,
        });
        // The wait hook is installed only for the async protocol (P2); the
        // synchronous feed leaves it NULL and the native keeps the old path.
        let (rows_fn, wait_fn, rows_ud): (
            ds4_sys::ds4_bridge_v41_rows_fn,
            ds4_sys::ds4_bridge_v41_wait_fn,
            *mut c_void,
        ) = match feed_tramp.as_mut() {
            Some(t) => {
                let ud = t as *mut FeedTramp<'_> as *mut c_void;
                if t.feed.async_protocol() {
                    (Some(rows_tramp), Some(wait_tramp), ud)
                } else {
                    (Some(rows_tramp), None, ud)
                }
            }
            None => (None, None, std::ptr::null_mut()),
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
                wait_fn,
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
