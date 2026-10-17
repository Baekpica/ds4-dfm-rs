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
use std::time::Instant;

use ds4_sys::{
    ds4_bridge_v41_generate, ds4_bridge_v41_images_bytes, ds4_bridge_v41_kvreuse_export,
    ds4_bridge_v41_kvreuse_import, ds4_bridge_v41_kvreuse_last_reused,
    ds4_bridge_v41_kvreuse_note_mount, ds4_bridge_v41_kvreuse_peek, ds4_bridge_v41_kvreuse_set,
    ds4_bridge_v41_last_spec_stats, ds4_bridge_v41_prefill_bytes, ds4_bridge_v41_set_dspark,
    ds4_bridge_v41_set_emit_trace, ds4_bridge_v41_set_graph, ds4_bridge_v41_set_prof,
    ds4_bridge_v41_set_sampling, ds4_bridge_v41_zchain_gr_set, ds4_bridge_v41_zchain_rb_clear,
    ds4_bridge_v41_zchain_rb_set,
};

use crate::engram::{EngramHash, EngramShard};
use crate::sidecar::{FNV_PRIME, FNV_SEED};
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

/// The last V4.1 run's restored token count (U-B): the engine's
/// `ds4_engine_v41_last_reused` (`core_v41_kvreuse.c:28`), which its server
/// reports as usage `cached_tokens` (`server_generate_v41.c:99-102`).  0 when
/// the request prefilled from token 0.
pub fn v41_kvreuse_last_reused() -> i32 {
    unsafe { ds4_bridge_v41_kvreuse_last_reused() }
}

/// The engine's `DS4_DEFAULT_MIN_P` (`ds4.h:54`, 0.02 since 2026-10-15,
/// commit 59ee755): what the V4.1 route uses when the request omits min_p.
/// 0.02 is the value that avoids both measured failure modes.  top_p 1.0 lets
/// the flat 2-bit tail through, dropping a junk token about every 160 (the
/// 2026-10-12 SVG).  top_p 0.95 -- and min_p 0.05, its twin -- collapse the
/// kept set to top-1 in a high-confidence repetition region, so `</think>`,
/// which the quantization presses down to 2-6%, can never be drawn and the
/// loop becomes absorbing (the 2026-10-15 28-minute stall).  min_p cuts at
/// 0.0198 relative to top-1: below the junk class (<9e-4), above `</think>`.
pub const V41_DEFAULT_MIN_P: f32 = 0.02;

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

/// The V4.1 disk half's host seam (item 4): the checkpoint store lives in
/// the Rust host (crates/ds4-kv, the port of ds4_kvstore.c), so the run
/// reaches it through this trait -- the same shape as `greedy`, which keeps
/// ds4-core store-agnostic.  The native owns the payload and the token
/// memcmp; the host owns the file, the budget and the eviction.
pub struct V41KvHit {
    /// The store entry's file.
    pub path: std::path::PathBuf,
    /// The payload range inside it (the store's envelope).
    pub offset: u64,
    pub bytes: u64,
    /// The matched token prefix (the store entry's token count).
    pub tokens: u32,
}

pub trait V41KvDisk {
    /// The longest token-prefix candidate strictly longer than `have` (the
    /// in-memory checkpoint's own prefix), already validated against the
    /// current mount digest and the payload ABI/class.  None = no candidate.
    fn find(&mut self, tokens: &[i32], sig: &[u64], digest: u64, have: u32) -> Option<V41KvHit>;
    /// Stage the native payload export (`save` writes it to a path the store
    /// owns) into a store entry.  Called only when the in-memory checkpoint
    /// is exactly this prompt; a failed `save` must leave no entry behind.
    fn store(
        &mut self,
        tokens: &[i32],
        sig: &[u64],
        digest: u64,
        save: &mut dyn FnMut(&Path) -> bool,
    );
}

/// A NUL-terminated copy of a path for the bridge calls; None on an interior
/// NUL (never for a real path).
fn c_path(path: &Path) -> Option<std::ffi::CString> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        std::ffi::CString::new(path.as_os_str().as_bytes()).ok()
    }
    #[cfg(not(unix))]
    {
        std::ffi::CString::new(path.to_str()?).ok()
    }
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
    /// `--no-kv-reuse` (U-B): lands 0 on the native's
    /// `ds4_engine_v41_set_kv_reuse` (the engine's server_config spelling);
    /// None leaves the native default (1, on).  The reference is
    /// `core_v41_kvreuse.c`; a cache hit must not change emitted ids, so the
    /// kill switch exists per unit policy.
    pub kv_reuse: Option<bool>,
    /// The V4.1 disk half (item 4): the KVC store handle when
    /// `--kv-disk-dir` opened one (crates/ds4-kv); None turns both halves
    /// off.  The native gates them on --no-kv-reuse and K.valid as well.
    pub kv_disk: Option<&'a mut dyn V41KvDisk>,
    /// The per-request sampling face (`ds4_engine_set_decode_sampling`,
    /// `core_v41_api.c:10-13`).  Some = set it before the run (the engine's
    /// server and CLI always do, `server_generate_v41.c:409` /
    /// `cli_diag.c:50`); None resets it to the all-zero face = bare argmax,
    /// so a run never inherits the previous request's face.
    pub sampling: Option<V41Sampling>,
    /// The tool-syntax greedy hook (the engine's ds4_greedy_fn, c6e2706): the
    /// serve's closure; the native asks once per position.  None = no hook.
    pub greedy: Option<&'a mut dyn FnMut(*const i32, u32) -> bool>,
    /// `--zchain <dir>`: the ② anti-repair sidecar (ds41_install.sh's own
    /// spelling; the deployment's third file next to the GGUF).  The engine
    /// carries it to the per-state mount (`core_engine_open.c:120-124` ->
    /// `v41_state_plugins`, `core_v41_state.c:77-87`); the port mounts it
    /// here, before the native call, with the same files read on the Rust
    /// host (`zchain.rs`, P2/P3).
    pub zchain: Option<&'a str>,
    /// `--posttrain <dir>`: the ③ post-train gain directory (the
    /// `base.fnv`-gated third file; finance-only in the deployment,
    /// ds41_install.sh).  Merged with ② exactly as `core_v41_amp.c:227-270`.
    pub posttrain: Option<&'a str>,
    /// The request's images (the engine's `ds4_v41_images`, core_v41.h:409):
    /// the Rust host preprocesses and expands the prompt, then borrows this in
    /// for the run.  None = a text request (the prefill never touches the
    /// sidecar).  The image identity is folded into the KV digest separately
    /// (see `kv_state_digest`).
    pub images: Option<&'a crate::v41_media::V41Images>,
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
            kv_reuse: None,
            kv_disk: None,
            sampling: None,
            greedy: None,
            zchain: None,
            posttrain: None,
            images: None,
        }
    }
}

/// The process-level asset caches (KU-1, 2026-10-13).
///
/// The engine opens the sidecar and the engram ONCE, when it opens the model
/// (`core_engine_open.c:120-124` -> `ds4_engine_v41_set_amp_dir`; the engram
/// shards at `core_v41_engram.c:218`), and every later request only generates.
/// The port re-did both on every request, which put the sidecar's read, its
/// device upload and its digest, plus a fresh engram open (GGUF, inventory,
/// wire, hash tables, shards, ring), on the first-token path of every decode -
/// the measured ~100 ms the port's warm floor carried over the engine's.
///
/// These caches hold what each open produced, keyed by the artifact identity:
/// a matching request reuses them, and any other key - or
/// `DS41_NO_SETUP_CACHE=1`, the A/B's control arm - recomputes them.  The
/// bare-base key still clears the sidecar, exactly as it did per request.
#[derive(Clone, PartialEq, Eq)]
struct MountKey {
    model: usize,
    zchain: Option<String>,
    posttrain: Option<String>,
}

#[derive(Clone, PartialEq, Eq)]
struct FeedKey {
    gguf: String,
    engram_dir: Option<String>,
}

/// The file-derived half of an open engram feed: everything that does not
/// depend on the prompt.  The engine keeps this for the life of its state.
#[derive(Clone)]
struct EngramAssets {
    core: Arc<Mutex<FeedCore>>,
    has_ring: bool,
    ring_cap: u32,
    cols: u32,
    stride: u32,
    n_layers: usize,
    /// The opened shard identities.  A hit must carry them: the checkpoint
    /// identity folds the engram mode and these sources in
    /// (`kv_state_digest`), so a hit without them would mint a different
    /// identity than the cold open did.
    sources: Vec<EngramSource>,
}

static ENGRAM_FEED: Mutex<Option<(FeedKey, EngramAssets)>> = Mutex::new(None);
static ZCHAIN_MOUNTED: Mutex<Option<(MountKey, i32, u64)>> = Mutex::new(None);

pub(crate) fn clear_setup_cache() {
    // The next model may reuse this native address. Its device mounts and
    // opened Engram sources must be established again, even at the same path.
    *ZCHAIN_MOUNTED
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = None;
    *ENGRAM_FEED
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = None;
}

impl EngramAssets {
    /// Take a built feed's asset half, for the cache.
    fn of(feed: &V41Feed) -> Self {
        Self {
            core: Arc::clone(&feed.core),
            has_ring: feed.has_ring,
            ring_cap: feed.ring_cap,
            cols: feed.cols,
            stride: feed.stride,
            n_layers: feed.n_layers,
            sources: feed.sources.clone(),
        }
    }
}

fn setup_cache_on() -> bool {
    std::env::var_os("DS41_NO_SETUP_CACHE").is_none()
}

/// Phase timings for the request prelude, `DS41_PHASE_LOG=1`.  The native
/// logs its own phases; this covers the host's, which is where the port's
/// warm floor lives.
fn phase_log(on: bool, what: &str, since: Instant) {
    if on {
        eprintln!(
            "ds4: [v41] host phase {}: {:.1} ms",
            what,
            since.elapsed().as_secs_f64() * 1e3
        );
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
    /// The declared next prefill block (the chunk loop's `prefetch_next`):
    /// the token range whose layer-0 rows the worker reads into the shadow
    /// after this block's last round.
    next: Option<(usize, usize)>,
    /// `DS41_FEED_NO_PREFETCH` (diagnostic): declarations are ignored, bit
    /// for bit the pre-prefetch behavior.
    prefetch_off: bool,
    /// The controlled-delay gate hooks (the P2 review's test): delay each
    /// layer's collect/publication by N us (a slow disk), or refuse every
    /// wait (the native route must then fail loud, never consume stale rows).
    /// Read once at open, so a run is one configuration.
    wait_delay_us: Option<u64>,
    wait_withhold: bool,
    sources: Vec<EngramSource>,
}

impl V41Feed {
    /// Open the feed from the artifact: the wire (with the `--engram-dir`
    /// rewrite), the four hash-constant tensors, and the shards.  None when
    /// the model carries no engram layers (the forward then takes no feed).
    /// Open the feed, or hand this request the one the last request opened.
    /// A hit skips the file work and rebuilds only the per-request state.
    pub fn open(path: &Path, engram_dir: Option<&str>) -> Result<Option<Self>> {
        let key = FeedKey {
            gguf: path.to_string_lossy().into_owned(),
            engram_dir: engram_dir.map(str::to_owned),
        };
        if setup_cache_on() {
            let hit = ENGRAM_FEED
                .lock()
                .unwrap()
                .as_ref()
                .filter(|(cached, _)| *cached == key)
                .map(|(_, assets)| assets.clone());
            if let Some(assets) = hit {
                eprintln!("ds4: [v41] engram rows: open cached (assets already open)");
                return Ok(Some(Self::from_assets(assets)));
            }
        }
        // A miss builds the whole open; its product is what the next request
        // with the same artifacts reuses.
        let feed = Self::open_uncached(path, engram_dir)?;
        if setup_cache_on() {
            if let Some(f) = &feed {
                *ENGRAM_FEED.lock().unwrap() = Some((key, EngramAssets::of(f)));
            }
        }
        Ok(feed)
    }

    /// The per-request state around an already-open asset set: the control
    /// block, the fetch worker, and a token history this prompt owns.  The
    /// engine's feed is per state, so its history starts empty at each open;
    /// clearing here is that same reset.  The shadow prefetch buffer is kept:
    /// it is overwritten before any read (its own comment above).
    fn from_assets(a: EngramAssets) -> Self {
        {
            let mut core = a.core.lock().unwrap();
            core.hist.clear();
            core.image_mask.clear();
        }
        let mut async_on = a.has_ring && std::env::var_os("DS41_FEED_SYNC").is_none();
        let ctl = Arc::new((
            Mutex::new(Ctl {
                go: false,
                done: 0,
                err: None,
                pf: None,
            }),
            Condvar::new(),
        ));
        let worker = if async_on {
            let (tx, rx) = mpsc::channel();
            let spawned = std::thread::Builder::new()
                .name("ds41-engram-fetch".into())
                .spawn({
                    let core = Arc::clone(&a.core);
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
        let prefetch_off = std::env::var_os("DS41_FEED_NO_PREFETCH").is_some();
        if prefetch_off && async_on {
            eprintln!("ds4: [v41] engram cross-block prefetch off (DS41_FEED_NO_PREFETCH)");
        }
        Self {
            core: a.core,
            worker,
            ctl,
            async_on,
            has_ring: a.has_ring,
            ring_cap: a.ring_cap,
            cols: a.cols,
            stride: a.stride,
            n_layers: a.n_layers,
            sources: a.sources.clone(),
            io_mode: false,
            prev_worker: false,
            started: vec![false; a.n_layers],
            joined: vec![false; a.n_layers],
            block_start: 0,
            block_end: 0,
            bases: Vec::new(),
            err: None,
            next: None,
            prefetch_off,
            wait_delay_us,
            wait_withhold,
        }
    }

    /// The asset half of the open, unchanged: this is the file work the cache
    /// above stores.
    fn open_uncached(path: &Path, engram_dir: Option<&str>) -> Result<Option<Self>> {
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
        let mut sources = Vec::with_capacity(wire.engram_layers.len());
        for path in &wire.engram_table_path {
            let shard = EngramShard::open(Path::new(path)).map_err(asset_error)?;
            sources.push(EngramSource {
                path: std::fs::canonicalize(path).map_err(asset_error)?,
                metadata: shard.metadata().map_err(asset_error)?,
            });
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
            image_mask: Vec::new(),
            shadow: Vec::new(),
        }));
        let ctl = Arc::new((
            Mutex::new(Ctl {
                go: false,
                done: 0,
                err: None,
                pf: None,
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
        let prefetch_off = std::env::var_os("DS41_FEED_NO_PREFETCH").is_some();
        if prefetch_off && async_on {
            eprintln!("ds4: [v41] engram cross-block prefetch off (DS41_FEED_NO_PREFETCH)");
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
            next: None,
            prefetch_off,
            wait_delay_us,
            wait_withhold,
            sources,
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

    /// U-B: seed the token history with a reused prompt prefix (the engine's
    /// `memcpy(st->hist, K.tok, K.n*4)`, `core_v41_kvreuse.c:118`).  The
    /// suffix's first block starts at the checkpoint position, so its n-gram
    /// window reads back into the prefix; the restore has already verified
    /// these tokens are this prompt's prefix.
    pub fn seed_prefix(&mut self, tokens: &[i32]) {
        let mut c = self.core.lock().unwrap();
        c.splice_hist(0, tokens);
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
        // Consume a completed cross-block prefetch: the worker read this
        // block's layer-0 rows into the shadow during the previous block's
        // tail.  Only an oversize block can have one (the chunk loop
        // declares prefill chunks only); a stale or mismatched shadow is
        // dropped and the round runs normally.
        let shadow_hit = {
            let (m, _) = &*self.ctl;
            let mut g = m.lock().unwrap();
            let hit = self.io_mode && g.pf == Some((pos0, end));
            g.pf = None;
            hit
        };
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
            if shadow_hit {
                // Hand the prefetched rows to the native's slot 0 (its own
                // pinned buffer; the previous block's reads of it are done
                // by the forward's per-block synchronize).
                let bytes = tokens.len() * self.cols as usize * self.stride as usize;
                unsafe { std::ptr::copy_nonoverlapping(c.shadow.as_ptr(), dst[0], bytes) };
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
                first: usize::from(shadow_hit),
                next: self.next.take(),
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
        self.next = None;
        for k in 0..self.n_layers {
            if !self.async_on {
                self.started[k] = true;
                self.joined[k] = true;
            } else if self.io_mode {
                self.started[k] = true;
                self.joined[k] = shadow_hit && k == 0;
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

    /// The cross-block prefetch declaration (the engine's
    /// `v41_engram_prefetch_next`, named at `core_v41_engram.c:230` and
    /// never implemented there): the chunk loop declares the NEXT prefill
    /// block before running the current one.  The declared block's tokens
    /// are spliced into the history here (the n-gram window reads them;
    /// consecutive blocks cover disjoint ranges, so a later block's tokens
    /// never change an earlier block's hash), and `prepare` chains the
    /// worker into reading the declared block's layer-0 round into the
    /// shadow after the current block's last round -- the ring is idle from
    /// there to the block end.  Only the first round is prefetched: the
    /// measured profile puts the stall at the first engram layer (L01
    /// +0.4 s at 2048, every later layer 0.0-0.1 s), and the second round
    /// already hides behind the layers between L1 and L14.
    ///
    /// A provider that cannot prefetch right now (the synchronous protocol,
    /// the kill switch, a block larger than the shadow) returns Ok and the
    /// next block simply runs without prefetch; Err is the sticky failure
    /// path and stops the run.
    pub fn prefetch_next(&mut self, pos0: u32, tokens: &[i32]) -> std::result::Result<(), String> {
        if let Some(e) = &self.err {
            return Err(e.clone());
        }
        if !self.async_on || self.prefetch_off || tokens.is_empty() {
            return Ok(());
        }
        let start = pos0 as usize;
        let end = start + tokens.len();
        let need = tokens.len() * self.cols as usize * self.stride as usize;
        let mut c = self.core.lock().unwrap();
        if c.shadow.len() < need {
            if !c.shadow.is_empty() {
                return Ok(()); // never grow: the worker may be mid-write
            }
            c.shadow.resize(need, 0);
        }
        c.splice_hist(start, tokens);
        drop(c);
        self.next = Some((start, end));
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

    /// End every write before the bridge frees its pinned destinations. Stop
    /// queues behind the active round; joining also covers a failed layer wait.
    fn drain(&mut self) -> std::result::Result<(), String> {
        let mut error = self.err.clone();
        if let Some(mut worker) = self.worker.take() {
            let _ = worker.tx.send(FeedCmd::Stop);
            if let Some(join) = worker.join.take() {
                if join.join().is_err() && error.is_none() {
                    error = Some("engram fetch worker panicked during drain".into());
                }
            }
        }
        let (ctl, _) = &*self.ctl;
        let ctl = ctl.lock().unwrap_or_else(|poison| poison.into_inner());
        if error.is_none() {
            error = ctl.err.clone();
        }
        drop(ctl);

        // Small/synchronous rounds have no worker. Their io_uring destinations
        // have the same lifetime, and a refusal may have skipped collect.
        let mut core = self
            .core
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(ring) = core.ring.as_mut() {
            if !ring.wait() && error.is_none() {
                error = Some("engram io_uring drain failed".into());
            }
        }
        self.joined.fill(true);
        match error {
            Some(error) => Err(error),
            None => Ok(()),
        }
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
    // Request positions are fixed before asynchronous prefetch starts. Draft
    // and decode positions beyond this mask remain ordinary text.
    image_mask: Vec<u8>,
    /// The cross-block prefetch buffer: one block's layer-0 rows in the
    /// native slot layout `[n][cols][stride]`, filled by the worker during
    /// the previous block's tail and copied into the native's slot 0 at the
    /// next prepare.  Sized once, at the first declaration, and never grown:
    /// only the last chunk is smaller than the cap, so no later block can
    /// need more, and a grow could move the buffer under the worker's
    /// in-flight reads.
    shadow: Vec<u8>,
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
            let rows = self
                .hash
                .rows_masked(&self.hist, &self.image_mask, p as i64, ei);
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
            let rows = self
                .hash
                .rows_masked(&self.hist, &self.image_mask, p as i64, ei);
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
    /// The shadow holds the layer-0 rows of the block whose `(pos0, end)`
    /// this is (a completed cross-block prefetch, waiting for its prepare).
    pf: Option<(u32, usize)>,
}

/// One oversize block for the worker (the engine's per-layer round loop,
/// `core_v41_engram.c:128-150`), optionally followed by the declared next
/// block's layer-0 round into the shadow (the cross-block prefetch).
enum FeedCmd {
    Rounds {
        start: usize,
        end: usize,
        bases: Vec<usize>,
        n_eng: usize,
        /// The first layer the worker runs: 1 when the shadow already holds
        /// layer 0 for this block (prepare copied it into the native slot).
        first: usize,
        /// The declared next block's token range: after this block's last
        /// round the worker reads its layer-0 rows into the shadow.
        next: Option<(usize, usize)>,
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
/// After the block's last round the declared next block's layer-0 round is
/// chained into the shadow (the cross-block prefetch).
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
        let (start, end, bases, n_eng, first, next) = match cmd {
            FeedCmd::Rounds {
                start,
                end,
                bases,
                n_eng,
                first,
                next,
            } => (start, end, bases, n_eng, first, next),
            FeedCmd::Stop => break,
        };
        let mut failed = false;
        for ei in first..n_eng {
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
        if !failed {
            if let Some((ns, ne)) = next {
                let r = catch_unwind(AssertUnwindSafe(|| {
                    let mut c = core.lock().unwrap();
                    let base = c.shadow.as_mut_ptr();
                    c.fill_round(0, ns, ne, base).and_then(|_| c.collect())
                }))
                .unwrap_or_else(|_| Err("engram fetch worker panicked".into()));
                let (m, cv) = &*ctl;
                let mut g = m.lock().unwrap();
                match r {
                    Ok(()) => {
                        g.pf = Some((ns as u32, ne));
                    }
                    Err(e) => {
                        if g.err.is_none() {
                            g.err = Some(format!("cross-block prefetch: {e}"));
                        }
                    }
                }
                cv.notify_all();
                drop(g);
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
        let _ = self.drain();
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

unsafe extern "C" fn drain_tramp(ud: *mut c_void) -> i32 {
    if ud.is_null() {
        return 1;
    }
    let t = &mut *(ud as *mut FeedTramp<'_>);
    match catch_unwind(AssertUnwindSafe(|| t.feed.drain())) {
        Ok(Ok(())) => 0,
        Ok(Err(_)) => 1,
        // Unwinding does not prove the writer ended. Do not let the bridge
        // release its destinations after an incomplete drain.
        Err(_) => std::process::abort(),
    }
}

/// The cross-block prefetch hook: the chunk loop declares the next prefill
/// block before running the current one; the feed reads its layer-0 round
/// during the current block's tail.  Same convention as prepare: nonzero
/// refuses the run (a provider error); "cannot prefetch right now" is Ok.
unsafe extern "C" fn next_tramp(pos0: u32, tokens: *const i32, n: i32, ud: *mut c_void) -> i32 {
    if ud.is_null() || n < 0 || tokens.is_null() {
        return 1;
    }
    if n == 0 {
        return 0;
    }
    let t = &mut *(ud as *mut FeedTramp<'_>);
    let tokens = std::slice::from_raw_parts(tokens, n as usize);
    match catch_unwind(AssertUnwindSafe(|| t.feed.prefetch_next(pos0, tokens))) {
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
/// bare execution: unload RB and GR before applying the requested tables.
/// This closes the oracle's omission when a gain layer disappears on a switch.
///
/// A directory carrying `amp_Lnn.bin` is refused by name: the low-rank
/// apply is not ported (the recorded §6.14 limit; no Spark directory
/// carries one). File validation precedes device writes; a failed upload
/// aborts the run, and the next mount clears all tables before retrying.
/// U-B's amplifier identity: FNV-1a-style folding over a float slice's bytes,
/// 8 bytes a step.  The native drops the KV checkpoint when the folded
/// identity of the mounted set changes (see `v41_mount_zchain`); a
/// deterministic change detector, not a standard FNV stream.
fn kv_mount_fold_f32s(h: &mut u64, data: &[f32]) {
    // SAFETY: every bit pattern of f32 is a valid read as bytes.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    kv_fold_bytes(h, bytes);
}

fn kv_fold_bytes(h: &mut u64, data: &[u8]) {
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        *h ^= w;
        *h = h.wrapping_mul(FNV_PRIME);
    }
    for &b in chunks.remainder() {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

fn v41_mount_zchain(model: &Model, zchain: Option<&str>, posttrain: Option<&str>) -> Result<u64> {
    // KU-1: the mount's product is device state, and nothing else in the
    // process changes it, so a request with the same artifact key reuses the
    // mount.  The caller still notes the per-request state identity, which
    // folds this mount digest together with the engram sources.
    let key = MountKey {
        model: model.raw_ptr() as usize,
        zchain: zchain.map(str::to_owned),
        posttrain: posttrain.map(str::to_owned),
    };
    if setup_cache_on() {
        let hit = ZCHAIN_MOUNTED
            .lock()
            .unwrap()
            .as_ref()
            .filter(|(cached, _, _)| *cached == key)
            .map(|(_, mode, digest)| (*mode, *digest));
        if let Some((mode, digest)) = hit {
            eprintln!("ds4: [v41] zchain mount: cached (identity {mode}/{digest})");
            return Ok(digest);
        }
    }
    let shape = model.shape();
    let geom = ZchainGeom {
        n_layer: shape.n_layer,
        n_expert: shape.n_expert,
        n_embd: shape.n_embd,
    };
    let merged = zchain_load_for_run(zchain, posttrain, geom)?;
    let mounted = merged.is_some();
    // Validation leaves the old mount intact; device writes may clear it
    // before an upload fails. Cache only a successfully completed mount.
    *ZCHAIN_MOUNTED.lock().unwrap() = None;
    let digest = apply_zchain(merged.as_ref(), geom, |write| match write {
        MountWrite::ClearRb => {
            unsafe { ds4_bridge_v41_zchain_rb_clear() };
            true
        }
        MountWrite::Bias(il, data) => unsafe {
            ds4_bridge_v41_zchain_rb_set(model.raw_ptr(), il, data.as_ptr(), geom.n_expert) != 0
        },
        MountWrite::Gain(il, data) => unsafe {
            ds4_bridge_v41_zchain_gr_set(
                model.raw_ptr(),
                il,
                data.map_or(std::ptr::null(), |data| data.as_ptr()),
                geom.n_expert,
                geom.n_embd,
            ) != 0
        },
    })?;
    if let Some(merged) = merged {
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
    }
    // KU-1: store the mount the next request with this key will reuse.  The
    // naked base (digest 0) is a mount state of its own and is cached too.
    if setup_cache_on() {
        *ZCHAIN_MOUNTED.lock().unwrap() = Some((key, i32::from(mounted), digest));
    }
    Ok(digest)
}

#[derive(Clone, Copy)]
enum EngramMode {
    On,
    Off,
}

/// Linux MemAvailable in MB (the engine's mem_available_mb,
/// server_sched_v41.c:22-29); None when /proc/meminfo cannot be read, which
/// leaves the admission open.  A negative value is not possible here.
fn mem_available_mb() -> Option<i64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: i64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

#[derive(Clone)]
struct EngramSource {
    path: std::path::PathBuf,
    metadata: std::fs::Metadata,
}

fn kv_state_digest(digest: u64, mode: EngramMode, sources: &[EngramSource]) -> Result<u64> {
    const KV_STATE_DOMAIN: &[u8] = b"v41-kv-state-v1";
    let mut digest = if digest == 0 { FNV_SEED } else { digest };
    kv_fold_bytes(&mut digest, KV_STATE_DOMAIN);
    // Image identity is position-specific, separately checked by native KV
    // and disk keys. A suffix image must not invalidate a safe text prefix.
    match mode {
        EngramMode::Off => {
            kv_fold_bytes(&mut digest, b"disabled");
            return Ok(digest);
        }
        EngramMode::On => kv_fold_bytes(&mut digest, b"enabled"),
    }
    // Local file-stat identity avoids reading two enormous Engram tables.
    // The stamp comes from the open shard, so it identifies the actual source.
    #[cfg(unix)]
    {
        use std::os::unix::{ffi::OsStrExt, fs::MetadataExt};
        for source in sources {
            let path = source.path.as_os_str().as_bytes();
            kv_fold_bytes(&mut digest, &(path.len() as u64).to_le_bytes());
            kv_fold_bytes(&mut digest, path);
            let m = &source.metadata;
            for field in [
                m.dev(),
                m.ino(),
                m.len(),
                m.mtime() as u64,
                m.mtime_nsec() as u64,
                m.ctime() as u64,
                m.ctime_nsec() as u64,
            ] {
                kv_fold_bytes(&mut digest, &field.to_le_bytes());
            }
        }
        Ok(digest)
    }
    #[cfg(not(unix))]
    {
        let _ = sources;
        Err(asset_error(
            "V4.1 Engram checkpoint identity requires Unix file metadata",
        ))
    }
}

enum MountWrite<'a> {
    ClearRb,
    Bias(u32, &'a [f32]),
    Gain(u32, Option<&'a [f32]>),
}

/// The mount order is shared by the native writes and the model-free device
/// mock. Loading/validating files finishes before any table changes.
fn apply_zchain(
    merged: Option<&V41Zchain>,
    geom: ZchainGeom,
    mut write: impl FnMut(MountWrite<'_>) -> bool,
) -> Result<u64> {
    write(MountWrite::ClearRb);
    for il in 0..geom.n_layer {
        if !write(MountWrite::Gain(il, None)) {
            return Err(asset_error(format!(
                "V4.1 zchain: gr L{il:02} unload refused"
            )));
        }
    }
    let Some(merged) = merged else {
        return Ok(0);
    };
    for (il, delta) in merged.rb.iter().enumerate() {
        let Some(delta) = delta else { continue };
        if !write(MountWrite::Bias(il as u32, delta)) {
            return Err(asset_error(format!(
                "V4.1 zchain: rb L{il:02} upload refused"
            )));
        }
    }
    for (il, gr) in merged.gr.iter().enumerate() {
        let Some(gr) = gr else { continue };
        if !write(MountWrite::Gain(il as u32, Some(&gr.factor))) {
            return Err(asset_error(format!(
                "V4.1 zchain: gr L{il:02} upload refused"
            )));
        }
    }
    // U-B: the checkpoint's KV was computed with the mounted amplifier, so the
    // native drops it when this identity changes (the engine's switch_plugins
    // drop, core_v41_state.c:111; the port mounts the same set per run, so
    // content identity stands in for the switch event).
    let mut digest: u64 = FNV_SEED;
    for (il, layer) in merged.rb.iter().enumerate() {
        digest ^= ((il as u64) << 32) | 1;
        digest = digest.wrapping_mul(FNV_PRIME);
        if let Some(rb) = layer {
            kv_mount_fold_f32s(&mut digest, rb);
        }
    }
    for (il, layer) in merged.gr.iter().enumerate() {
        digest ^= ((il as u64) << 32) | 2;
        digest = digest.wrapping_mul(FNV_PRIME);
        if let Some(gr) = layer {
            kv_mount_fold_f32s(&mut digest, &gr.factor);
        }
    }
    Ok(digest)
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
        if let Some(images) = opts.images {
            images.validate(prompt)?;
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
            // U-B: --no-kv-reuse only; None leaves the native default (on).
            if let Some(on) = opts.kv_reuse {
                ds4_bridge_v41_kvreuse_set(i32::from(on));
            }
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
        // the engine's per-state mount inside the generate. Checkpoints also
        // bind the Engram mode and the actual opened sources, before any peek
        // or disk import; equal prompt tokens alone are insufficient.
        //
        // KU-1: both opens are cached per process, so the phase log is what
        // shows their cost; a hit prints its own cached line instead.
        let phases = std::env::var_os("DS41_PHASE_LOG").is_some();
        let t_prelude = Instant::now();
        let t_feed = Instant::now();
        let mut feed = if opts.no_engram {
            None
        } else {
            V41Feed::open(path, opts.engram_dir)?
        };
        let signatures = opts.images.map_or(&[][..], |i| i.sig_of.as_slice());
        let sig_ptr = if signatures.is_empty() {
            std::ptr::null()
        } else {
            signatures.as_ptr()
        };
        if let Some(f) = &feed {
            let mut core = f.core.lock().unwrap();
            core.image_mask.clear();
            core.image_mask
                .extend(signatures.iter().map(|&sig| u8::from(sig != 0)));
        }
        phase_log(phases, "engram feed open", t_feed);
        let t_mount = Instant::now();
        let sidecar = v41_mount_zchain(self, opts.zchain, opts.posttrain)?;
        phase_log(phases, "sidecar mount", t_mount);
        let digest = kv_state_digest(
            sidecar,
            if opts.no_engram {
                EngramMode::Off
            } else {
                EngramMode::On
            },
            feed.as_ref().map_or(&[], |feed| &feed.sources),
        )?;
        unsafe { ds4_bridge_v41_kvreuse_note_mount(1, digest) };
        // The disk half (item 4): the find runs after the mount so the
        // candidate's trailer is checked against THIS run's state
        // identity; only a candidate strictly longer than the in-memory
        // checkpoint is imported (a shorter one would regress the reuse K
        // already gives).  The store's match only selects -- the native
        // import re-validates sizes and the restore re-runs the token memcmp.
        if let Some(disk) = opts.kv_disk.as_mut() {
            let have = unsafe {
                ds4_bridge_v41_kvreuse_peek(prompt.as_ptr(), prompt.len() as i32, sig_ptr)
            };
            if let Some(hit) = disk.find(prompt, signatures, digest, have) {
                if let Some(cpath) = c_path(&hit.path) {
                    let ok = unsafe {
                        ds4_bridge_v41_kvreuse_import(
                            cpath.as_ptr(),
                            hit.offset,
                            hit.bytes,
                            prompt.as_ptr(),
                            hit.tokens,
                        )
                    };
                    if ok == 0 {
                        eprintln!(
                            "ds4: [v41] KV disk checkpoint did not load; the request continues without it"
                        );
                    }
                }
            }
        }
        // U-B's host seam (ds4_ds41_kvreuse.inc): the native checkpoint owns
        // the reuse decision, but the engram history lives here, and the hash
        // reads back three positions, so the feed is seeded with the tokens
        // the restore will adopt (the engine's st->hist memcpy,
        // core_v41_kvreuse.c:118).  The native restore re-runs the memcmp, so
        // a stale peek only ever seeds this prompt's own prefix.
        if let Some(f) = feed.as_mut() {
            let reused = unsafe {
                ds4_bridge_v41_kvreuse_peek(prompt.as_ptr(), prompt.len() as i32, sig_ptr)
            };
            if reused > 0 {
                f.seed_prefix(&prompt[..reused as usize]);
            }
        }
        // The tramp borrows the feed for the duration of the call only; the
        // native never retains the pointer past it (ds4_bridge.h).
        let mut feed_tramp = feed.as_mut().map(|f| FeedTramp {
            n_layers: f.n_layers,
            feed: f,
        });
        // The wait hook is installed only for the async protocol (P2); the
        // synchronous feed leaves it NULL and the native keeps the old path.
        // The prefetch hook rides the same condition: without the worker
        // there is no ring to chain the next block's round into.
        let (rows_fn, wait_fn, next_fn, rows_ud): (
            ds4_sys::ds4_bridge_v41_rows_fn,
            ds4_sys::ds4_bridge_v41_wait_fn,
            ds4_sys::ds4_bridge_v41_next_fn,
            *mut c_void,
        ) = match feed_tramp.as_mut() {
            Some(t) => {
                let ud = t as *mut FeedTramp<'_> as *mut c_void;
                if t.feed.async_protocol() {
                    (Some(rows_tramp), Some(wait_tramp), Some(next_tramp), ud)
                } else {
                    (Some(rows_tramp), None, None, ud)
                }
            }
            None => (None, None, None, std::ptr::null_mut()),
        };
        phase_log(phases, "host prelude (mount + open + seed)", t_prelude);
        let mut emit_tramp_state = EmitTramp { emit };
        let mut progress_tramp_state = ProgressTramp { progress };
        let mut err = [0u8; 512];
        // Vision: build the bridge image view from the prepared set (borrowed
        // for the call).  Empty = a text request; pass NULL.
        let bridge_raw: Vec<ds4_sys::ds4_bridge_v41_image> = match opts.images {
            Some(imgs) => imgs
                .images
                .iter()
                .map(|im| ds4_sys::ds4_bridge_v41_image {
                    patches: im.patches.as_ptr(),
                    patches_len: im.patches.len() as u64,
                    n_vit_h: im.plan.n_vit_h,
                    n_vit_w: im.plan.n_vit_w,
                    n_llm_h: im.plan.n_llm_h,
                    n_llm_w: im.plan.n_llm_w,
                })
                .collect(),
            None => Vec::new(),
        };
        let bridge_images = ds4_sys::ds4_bridge_v41_images {
            img: bridge_raw.as_ptr(),
            n: bridge_raw.len() as i32,
            row_of: opts.images.map_or(std::ptr::null(), |i| i.row_of.as_ptr()),
            sig_of: sig_ptr,
            np: opts.images.map_or(0, |i| i.np),
            n_rows: opts.images.map_or(0, |i| i.n_rows),
        };
        let images_ptr: *const ds4_sys::ds4_bridge_v41_images = if bridge_raw.is_empty() {
            std::ptr::null()
        } else {
            &bridge_images
        };
        // The serving admission (server_sched_v41.c:58): start the prefill only
        // when MemAvailable covers twice its peak (the engine's margin for
        // backend scratch and the index draft's transient); otherwise refuse
        // this request rather than risk an OOM.
        {
            let peak = unsafe { ds4_bridge_v41_prefill_bytes(self.raw_ptr(), prompt.len() as i32) }
                + unsafe { ds4_bridge_v41_images_bytes(images_ptr) };
            let peak_mb = peak >> 20;
            if let Some(avail_mb) = mem_available_mb() {
                if avail_mb >= 0 && (avail_mb as u64) < 2u64 * peak_mb {
                    return Err(asset_error(&format!(
                        "prefill needs {peak_mb} MB peak (admission 2x), only {avail_mb} MB available"
                    )));
                }
            }
        }
        let rc = unsafe {
            ds4_bridge_v41_generate(
                self.raw_ptr(),
                prompt.as_ptr(),
                prompt.len() as i32,
                n_predict,
                i32::from(opts.no_engram),
                opts.verify_k,
                images_ptr,
                rows_fn,
                rows_ud,
                wait_fn,
                next_fn,
                feed_tramp
                    .as_ref()
                    .map(|_| drain_tramp as unsafe extern "C" fn(*mut c_void) -> i32),
                Some(emit_tramp),
                &mut emit_tramp_state as *mut EmitTramp<'_> as *mut c_void,
                Some(progress_tramp),
                &mut progress_tramp_state as *mut ProgressTramp<'_> as *mut c_void,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        // The disk export (item 4): only when the in-memory checkpoint is
        // exactly this prompt -- the native save runs at this prompt's
        // prefill end, so K.n == np means K is this request's checkpoint
        // (peek returns K.n under the same memcmp the restore uses).  A
        // decode abort does not invalidate it; a prefill abort leaves the
        // previous, shorter checkpoint and stores nothing.  The store's own
        // reuse check skips an entry already current.
        if let Some(disk) = opts.kv_disk.as_mut() {
            let have = unsafe {
                ds4_bridge_v41_kvreuse_peek(prompt.as_ptr(), prompt.len() as i32, sig_ptr)
            };
            if have == prompt.len() as u32 {
                let mut save = |p: &Path| -> bool {
                    match c_path(p) {
                        Some(c) => unsafe { ds4_bridge_v41_kvreuse_export(c.as_ptr()) != 0 },
                        None => false,
                    }
                };
                disk.store(prompt, signatures, digest, &mut save);
            }
        }
        if rc != 0 {
            return Err(crate::fail(rc, &err));
        }
        Ok(())
    }

    /// KU-2: pay the V4.1 asset opens for the artifacts the first request will
    /// use, so request one does not.  The engine pays both at model open
    /// (`core_engine_open.c:120-124` -> `ds4_engine_v41_set_amp_dir`; the
    /// engram shards at `core_v41_engram.c:218`); the port pays them inside
    /// request one, which is the one cell the engine still wins.
    ///
    /// Both calls are the prelude's own (`v41_mount_zchain`, `V41Feed::open`),
    /// with the same arguments, so the caches fill with the keys request one
    /// will look up.  `DS41_NO_SETUP_CACHE=1` disables the whole thing: with
    /// the caches off there is nothing to fill, and the work would be paid
    /// twice.
    pub fn v41_prewarm(
        &self,
        path: &Path,
        zchain: Option<&str>,
        posttrain: Option<&str>,
        engram_dir: Option<&str>,
        no_engram: bool,
    ) -> Result<()> {
        if !setup_cache_on() {
            return Ok(());
        }
        let phases = std::env::var_os("DS41_PHASE_LOG").is_some();
        let t_mount = Instant::now();
        v41_mount_zchain(self, zchain, posttrain)?;
        phase_log(phases, "prewarm sidecar mount", t_mount);
        if !no_engram {
            let t_feed = Instant::now();
            // Only the asset half is wanted here; the returned feed is the
            // per-request state, and ENGRAM_FEED keeps the half that lasts.
            let feed = V41Feed::open(path, engram_dir)?;
            drop(feed);
            phase_log(phases, "prewarm engram feed open", t_feed);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static CACHE_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Default)]
    struct MountMock {
        gains: Vec<Option<Vec<f32>>>,
        writes: usize,
        fail_upload: Option<u32>,
    }

    static MOUNT_DEVICE: Mutex<MountMock> = Mutex::new(MountMock {
        gains: Vec::new(),
        writes: 0,
        fail_upload: None,
    });

    #[export_name = "ds4_bridge_v41_zchain_gr_set"]
    unsafe extern "C" fn mock_gr_set(
        _: *mut ds4_sys::ds4_bridge_model,
        il: u32,
        data: *const f32,
        n_expert: u32,
        n_embd: u32,
    ) -> i32 {
        let mut device = MOUNT_DEVICE.lock().unwrap();
        device.writes += 1;
        if il as usize >= device.gains.len()
            || n_expert != GEOM.n_expert
            || n_embd != GEOM.n_embd
            || (!data.is_null() && device.fail_upload == Some(il))
        {
            return 0;
        }
        device.gains[il as usize] = if data.is_null() {
            None
        } else {
            Some(unsafe { std::slice::from_raw_parts(data, (n_expert * n_embd) as usize) }.to_vec())
        };
        1
    }

    #[export_name = "ds4_bridge_v41_zchain_rb_clear"]
    extern "C" fn mock_rb_clear() {
        MOUNT_DEVICE.lock().unwrap().writes += 1;
    }

    #[export_name = "ds4_bridge_v41_zchain_rb_set"]
    extern "C" fn mock_rb_set(
        _: *mut ds4_sys::ds4_bridge_model,
        _: u32,
        _: *const f32,
        _: u32,
    ) -> i32 {
        0 // These fixtures carry gain tables only.
    }

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

    fn drop_fixture(path: &Path) -> Model {
        use crate::{Backend, BindPlan, ModelFamily, Variant, Vocab};

        let tokens = [
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
            "｜DSML｜",
        ];
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        for (key, values) in [
            ("tokenizer.ggml.tokens", tokens.as_slice()),
            ("tokenizer.ggml.merges", [].as_slice()),
        ] {
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(&crate::gguf::GGUF_VALUE_ARRAY.to_le_bytes());
            bytes.extend_from_slice(&crate::gguf::GGUF_VALUE_STRING.to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
                bytes.extend_from_slice(value.as_bytes());
            }
        }
        std::fs::write(path, bytes).unwrap();
        let inventory = TensorInventory::open(path).unwrap();
        let bind_plan = BindPlan {
            shape: crate::shape_for_variant(Variant::DeepSeek41Flash),
            slots: Vec::new(),
            n_shards: 1,
            data_pos: inventory.data_pos,
            alignment: inventory.alignment,
            page: inventory.page,
        };
        Model {
            raw: std::ptr::NonNull::dangling(),
            family: ModelFamily::DeepSeek4,
            backend: Backend::Cuda,
            inventory,
            bind_plan,
            mtp: None,
            dspark: None,
            vocab: Vocab::load_path(path, ModelFamily::DeepSeek4).unwrap(),
            chat_template: None,
            vision_ready: false,
            mtp_draft_tokens: 0,
            runtime_budget: None,
            v41_ctx: Some(4096),
            _distributed: None,
            _not_send: std::marker::PhantomData,
        }
    }

    #[test]
    fn model_drop_evicts_assets() {
        let _serial = CACHE_TEST_LOCK.lock().unwrap();
        let dir = tmpdir("model-drop");
        // The native free is stubbed by lib.rs tests. Reusing the same handle
        // reproduces allocator address reuse without loading device weights.
        for _ in 0..2 {
            let model = drop_fixture(&dir.join("tiny.gguf"));
            let core = Arc::new(Mutex::new(FeedCore {
                ring: None,
                hash: EngramHash {
                    token_map: Vec::new(),
                    multipliers: Vec::new(),
                    primes: Vec::new(),
                    offsets: Vec::new(),
                    max_ngram: 2,
                    heads: 1,
                    pad: 0,
                    n_vocab: 0,
                },
                shards: Vec::new(),
                weight_off: Vec::new(),
                scale_off: Vec::new(),
                rows: Vec::new(),
                head_dim: 0,
                cols: 0,
                stride: 0,
                hist: Vec::new(),
                image_mask: Vec::new(),
                shadow: Vec::new(),
            }));
            let weak = Arc::downgrade(&core);
            *ENGRAM_FEED.lock().unwrap() = Some((
                FeedKey {
                    gguf: "old-model.gguf".into(),
                    engram_dir: None,
                },
                EngramAssets {
                    core,
                    has_ring: false,
                    ring_cap: 0,
                    cols: 0,
                    stride: 0,
                    n_layers: 0,
                    sources: Vec::new(),
                },
            ));
            *ZCHAIN_MOUNTED.lock().unwrap() = Some((
                MountKey {
                    model: model.raw_ptr() as usize,
                    zchain: Some("mounted".into()),
                    posttrain: None,
                },
                1,
                42,
            ));
            drop(model);
            assert!(
                ZCHAIN_MOUNTED.lock().unwrap().is_none(),
                "closed model left a cached device mount for its next handle owner"
            );
            assert!(
                weak.upgrade().is_none(),
                "closed model kept its cached Engram sources alive"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn mount_fixture(dir: &Path) -> Model {
        let mut model = drop_fixture(&dir.join("tiny.gguf"));
        model.bind_plan.shape.n_layer = GEOM.n_layer;
        model.bind_plan.shape.n_expert = GEOM.n_expert;
        model.bind_plan.shape.n_embd = GEOM.n_embd;
        *ZCHAIN_MOUNTED.lock().unwrap() = None;
        *MOUNT_DEVICE.lock().unwrap() = MountMock {
            gains: vec![None; GEOM.n_layer as usize],
            ..MountMock::default()
        };
        model
    }

    #[test]
    fn failed_mount_reloads_old() {
        let _serial = CACHE_TEST_LOCK.lock().unwrap();
        let dir = tmpdir("mount-failure");
        let a = dir.join("a");
        let b = dir.join("b");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(a.join("gr_L00.bin"), gr_f32(4, 2)).unwrap();
        std::fs::write(b.join("gr_L01.bin"), gr_f32(4, 2)).unwrap();
        let model = mount_fixture(&dir);
        let digest = v41_mount_zchain(&model, a.to_str(), None).unwrap();
        let expected = MOUNT_DEVICE.lock().unwrap().gains.clone();
        MOUNT_DEVICE.lock().unwrap().fail_upload = Some(1);

        let error = v41_mount_zchain(&model, b.to_str(), None).unwrap_err();
        assert!(error.message.contains("gr L01 upload refused"));
        let invalidated = ZCHAIN_MOUNTED.lock().unwrap().is_none();
        let writes = MOUNT_DEVICE.lock().unwrap().writes;
        MOUNT_DEVICE.lock().unwrap().fail_upload = None;

        // B cleared A before its upload failed. Retrying A must restore its
        // device tables, not return A's stale cached identity.
        assert_eq!(v41_mount_zchain(&model, a.to_str(), None).unwrap(), digest);
        let (actual, retried) = {
            let device = MOUNT_DEVICE.lock().unwrap();
            (device.gains.clone(), device.writes > writes)
        };
        assert_eq!(actual, expected, "retry used a stale mount cache hit");
        assert!(retried, "retry did not upload the original gain table");
        assert!(invalidated, "failed mutation retained a cache identity");
        drop(model);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_mount_keeps_cache() {
        let _serial = CACHE_TEST_LOCK.lock().unwrap();
        let dir = tmpdir("mount-validation");
        let a = dir.join("a");
        let invalid = dir.join("invalid");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&invalid).unwrap();
        std::fs::write(a.join("gr_L00.bin"), gr_f32(4, 2)).unwrap();
        let model = mount_fixture(&dir);
        let digest = v41_mount_zchain(&model, a.to_str(), None).unwrap();
        let writes = MOUNT_DEVICE.lock().unwrap().writes;

        let error = v41_mount_zchain(&model, invalid.to_str(), None).unwrap_err();
        assert!(error.message.contains("zchain-empty"));
        assert_eq!(MOUNT_DEVICE.lock().unwrap().writes, writes);
        assert_eq!(v41_mount_zchain(&model, a.to_str(), None).unwrap(), digest);
        assert_eq!(MOUNT_DEVICE.lock().unwrap().writes, writes);
        drop(model);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn no_dirs_answers_none() {
        assert!(zchain_load_for_run(None, None, GEOM).unwrap().is_none());
    }

    fn mock_mount(merged: Option<&V41Zchain>, gains: &mut [Option<Vec<f32>>]) -> u64 {
        apply_zchain(merged, GEOM, |write| {
            if let MountWrite::Gain(il, data) = write {
                gains[il as usize] = data.map(|data| data.to_vec());
            }
            true
        })
        .unwrap()
    }

    #[test]
    fn mount_switch_clears_old_gains() {
        let a = tmpdir("switch-a");
        let b = tmpdir("switch-b");
        std::fs::write(a.join("gr_L00.bin"), gr_f32(4, 2)).unwrap();
        std::fs::write(b.join("gr_L01.bin"), gr_f32(4, 2)).unwrap();
        let a = zchain_load_for_run(a.to_str(), None, GEOM).unwrap();
        let b = zchain_load_for_run(b.to_str(), None, GEOM).unwrap();
        let mut gains = vec![None; GEOM.n_layer as usize];
        mock_mount(a.as_ref(), &mut gains);
        mock_mount(b.as_ref(), &mut gains);
        assert!(
            gains[0].is_none(),
            "old mount's absent layer remained active"
        );
        assert!(gains[1].is_some());
    }

    #[test]
    fn bare_mount_clears_all_gains() {
        let dir = tmpdir("bare-switch");
        std::fs::write(dir.join("gr_L00.bin"), gr_f32(4, 2)).unwrap();
        let mounted = zchain_load_for_run(dir.to_str(), None, GEOM).unwrap();
        let mut gains = vec![None; GEOM.n_layer as usize];
        mock_mount(mounted.as_ref(), &mut gains);
        assert_eq!(mock_mount(None, &mut gains), 0);
        assert!(
            gains.iter().all(Option::is_none),
            "bare digest kept mounted gains"
        );
    }

    #[test]
    fn engram_mode_changes_state_key() {
        let dir = tmpdir("engram-mode");
        let source = dir.join("source.safetensors");
        std::fs::write(&source, b"engram").unwrap();
        let sources = [source_stamp(&source)];
        let enabled = kv_state_digest(7, EngramMode::On, &sources).unwrap();
        let disabled = kv_state_digest(7, EngramMode::Off, &sources).unwrap();
        assert_ne!(
            enabled, disabled,
            "Engram-disabled requests reused enabled KV"
        );
    }

    #[test]
    fn engram_source_changes_key() {
        let dir = tmpdir("engram-source");
        let a = dir.join("a.safetensors");
        let b = dir.join("b.safetensors");
        std::fs::write(&a, b"engram-a").unwrap();
        std::fs::write(&b, b"engram-b").unwrap();
        let before = kv_state_digest(7, EngramMode::On, &[source_stamp(&a)]).unwrap();
        assert_eq!(
            before,
            kv_state_digest(7, EngramMode::On, &[source_stamp(&a)]).unwrap()
        );
        assert_ne!(
            before,
            kv_state_digest(7, EngramMode::On, &[source_stamp(&b)]).unwrap(),
            "a different source path reused old Engram KV"
        );
        std::fs::rename(b, &a).unwrap();
        assert_ne!(
            before,
            kv_state_digest(7, EngramMode::On, &[source_stamp(&a)]).unwrap(),
            "a replaced source reused old Engram KV"
        );
    }

    #[test]
    fn opened_source_keeps_its_stamp() {
        let dir = tmpdir("open-source");
        let path = dir.join("source.safetensors");
        let replacement = dir.join("replacement.safetensors");
        std::fs::write(&path, b"old").unwrap();
        std::fs::write(&replacement, b"new").unwrap();
        let shard = EngramShard::open(&path).unwrap();
        std::fs::rename(replacement, &path).unwrap();
        let opened = EngramSource {
            path: path.canonicalize().unwrap(),
            metadata: shard.metadata().unwrap(),
        };
        assert_ne!(
            kv_state_digest(7, EngramMode::On, &[opened]).unwrap(),
            kv_state_digest(7, EngramMode::On, &[source_stamp(&path)]).unwrap(),
            "checkpoint identity followed the pathname instead of its open source"
        );
    }

    fn source_stamp(path: &Path) -> EngramSource {
        let shard = EngramShard::open(path).unwrap();
        EngramSource {
            path: path.canonicalize().unwrap(),
            metadata: shard.metadata().unwrap(),
        }
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

    /// The prewarm runs only when the caches are consulted at all: with
    /// `DS41_NO_SETUP_CACHE=1` the same opens would be paid twice (once here,
    /// once in request one), which is the behaviour KU-2 must not introduce.
    #[test]
    fn prewarm_gate_follows_the_kill_switch() {
        let _serial = CACHE_TEST_LOCK.lock().unwrap();
        let prior = std::env::var_os("DS41_NO_SETUP_CACHE");
        std::env::set_var("DS41_NO_SETUP_CACHE", "1");
        assert!(
            !setup_cache_on(),
            "the kill switch must close the prewarm gate"
        );
        match prior {
            Some(value) => std::env::set_var("DS41_NO_SETUP_CACHE", value),
            None => std::env::remove_var("DS41_NO_SETUP_CACHE"),
        }
        assert!(setup_cache_on(), "an unset switch must open the gate");
    }
}
