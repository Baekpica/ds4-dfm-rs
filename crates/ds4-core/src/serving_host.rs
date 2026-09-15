//! Host adapter that fills serving-plan quote facts without opening an engine.
//!
//! Weights are the mapped GGUF span. Per-bank / scratch / checkpoint / PLE /
//! media are shape quotes. Available memory is the live host observation.

use std::path::{Path, PathBuf};

use crate::gguf::GgufFile;
use crate::serving::{
    BankLane, EngineFacts, LaneMode, MaxSeqs, MtpKind, MtpMode, PrefixReuse, ReuseKind,
    ServingCaps, ServingRequest, DEFAULT_MAX_SEQS, DEFAULT_SCHED_CHUNK,
};
use crate::shape::{ModelFamily, Shape, Variant};
use crate::tensors::model_split_sibling_path;
use crate::Backend;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
const DEFAULT_PLE_CACHE_MB: u64 = 2048;
const PLE_CACHE_MB_ENV: &str = "DS4_QWEN_PLE_CACHE_MB";
const PLE_CACHE_MB_512: u64 = 512;
const PLE_CACHE_MB_1024: u64 = 1024;
const QWEN_PREFILL_CHUNK_ENV: &str = "DS4_QWEN_PREFILL_CHUNK";
const STEP_PREFILL_CHUNK_ENV: &str = "DS4_STEP37_PREFILL_CHUNK";
const INKLING_PREFILL_CHUNK_ENV: &str = "DS4_INKLING_PREFILL_CHUNK";
const QWEN_NATIVE_DEFAULT: u32 = 256;
const QWEN_NATIVE_MAX: u32 = 16384;
const STEP_NATIVE_DEFAULT: u32 = 4096;
const STEP_NATIVE_MAX: u32 = 4096;
const INKLING_NATIVE_DEFAULT: u32 = 1024;
const INKLING_NATIVE_MAX: u32 = 8192;
const INKLING_REL_DIM: u64 = 16;
const INKLING_GLOBAL_ROWS: u64 = 1024;
const INKLING_DRAFT_GLOBALS: u64 = 2;
const STEP_VISION_EDGE: u64 = 728;
const STEP_VISION_PATCH: u64 = 14;
const STEP_VISION_DIM: u64 = 1536;
const STEP_VISION_FFN: u64 = 8960;
const STEP_MEDIA_ROWS: u64 = 8192;
const GLM_NATIVE_DEFAULT: u32 = 2048;
const GLM_ATTENTION_PERIOD: u32 = 4;
const EXAONE_PREFILL_CHUNK_ENV: &str = "DS4_EXAONE_PREFILL_CHUNK";
const EXAONE_NATIVE_DEFAULT: u32 = 512;
const K2_NATIVE_DEFAULT: u32 = 1024;
const MOTIF_PREFILL_CHUNK_ENV: &str = "DS4_MOTIF3_PREFILL_CHUNK";
const MOTIF_NATIVE_DEFAULT: u32 = 4096;
const MOTIF_NATIVE_MAX: u32 = 8192;
const SOLAR_PREFILL_CHUNK_ENV: &str = "DS4_METAL_PREFILL_CHUNK";
const SOLAR_NATIVE_DEFAULT: u32 = 2048;
const SOLAR_KV_FORMAT_ENV: &str = "DS4_SOLAR_KV_FORMAT";
const FIT_HEADROOM_ENV: &str = "DS4_BATCH_FIT_HEADROOM_MB";
const FIT_DERIVED_ENV: &str = "DS4_BATCH_FIT_HEADROOM_DERIVED";
const FIT_BURST_ENV: &str = "DS4_BATCH_FIT_BURST_MB";
const FIT_STATIC_MB: u64 = 6144;
const FIT_BURST_MB: u64 = 2048;
const SESSION_FIT_ENV: &str = "DS4_SESSION_GRAPH_FIT";
const SESSION_HEADROOM_ENV: &str = "DS4_SESSION_GRAPH_HEADROOM_MB";
const SESSION_HEADROOM_MB: u64 = 1024;
const DOTS3_PREFILL_CHUNK_ENV: &str = "DS4_DOTS3_PREFILL_CHUNK";
const DOTS3_NATIVE_DEFAULT: u32 = 4096;
const DOTS3_NATIVE_MAX: u32 = 8192;
const DOTS3_INDEX_ROWS: u64 = 128;
const DOTS3_PARTIAL_ROWS: u64 = 2;
const DOTS3_PARTIAL_SPLITS: u64 = 16;
const FAMILY_NATIVE_MAX: u32 = 16384;
const QWEN_QSA_NO_FUSED_ENV: &str = "DS4_QWEN_QSA_NO_FUSED";
const WEIGHT_IPC_MANIFEST_ENV: &str = "DS4_CUDA_WEIGHT_IPC_MANIFEST";
const WEIGHT_IPC_SCOPE_ENV: &str = "DS4_CUDA_WEIGHT_IPC_SCOPE";
const QWEN_GRAPH_LOWRANK: u64 = 320;
const QWEN_GRAPH_RATIO: u64 = 4;
const QWEN_GRAPH_SELECTED_BLOCKS_MAX: u64 = 512;
const QWEN_GDN_KEY_HEADS: u64 = 16;
const QWEN_GDN_VALUE_HEADS: u64 = 48;
const QWEN_GDN_HEAD: u64 = 128;
const QWEN_QSA_SCORE_ROWS: u32 = 8;
const QWEN_PLE_HOST_ID_LANES: u64 = 16;
const QWEN_PLE_CONV_TAPS: u64 = 9;
const CHECKPOINT_SLOTS: u64 = 32;
const SIZEOF_F32: u64 = 4;
const SIZEOF_I32: u64 = 4;
const SIZEOF_U16: u64 = 2;
const SIZEOF_U32: u64 = 4;
const SIZEOF_U64: u64 = 8;

/// What the host can gather before `Model::open`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuoteHost {
    pub weights_bytes: u64,
    pub mtp_bytes: u64,
    pub available_bytes: u64,
    pub native_chunk: Option<u32>,
    pub vision: bool,
}

/// Map the named P2 budgets onto `EngineFacts` so `resolve_plan` can quote.
pub fn fill_quote_facts(
    facts: &mut EngineFacts,
    req: &ServingRequest,
    caps: ServingCaps,
    shape: Option<Shape>,
    host: QuoteHost,
) {
    // C honors family env (DS4_QWEN_PREFILL_CHUNK), not --native-chunk.
    // A quoted cap above the rows C will allocate lets --prefill-chunk 512
    // pass while Qwen still builds 256.
    let ctx_tokens = req.ctx.max(1) as u32;
    let runtime = family_native_chunk(caps, ctx_tokens);
    let mut native = host
        .native_chunk
        .or(req.native_chunk)
        .unwrap_or(runtime)
        .min(runtime)
        .max(1);
    // Step/Inkling create predictor state whenever the sidecar is loaded, even
    // with speculation disabled. Its verify workspace needs draft layers + 1.
    let sidecar_loaded = req.mtp_path.is_some() || facts.mtp_loaded;
    let step_mtp = caps.family == ModelFamily::Step37 && sidecar_loaded;
    if sidecar_loaded && matches!(caps.family, ModelFamily::Step37 | ModelFamily::Inkling) {
        if let Some(s) = shape {
            native = native.max(ctx_tokens.min(s.n_nextn_predict + 1));
        }
    }
    let ctx = u64::from(ctx_tokens);
    let kv = shape.map(|s| bank_kv_bytes(s, ctx, native)).unwrap_or(0);
    // Other families gate speculative allocations on execution settings.
    // Without a bank driver the host does not publish an auto Qwen draft.
    // Explicit drafts still reach native open; its default is one.
    let default_draft = if caps.mtp == MtpKind::Embedded && !quote_bank_lane(req, caps, facts) {
        1
    } else {
        caps.spec_draft_min
    };
    let draft = req.mtp_draft.unwrap_or(default_draft);
    let mtp_on = match caps.mtp {
        MtpKind::Embedded => req.mtp_mode != MtpMode::Off && draft >= caps.spec_draft_min,
        MtpKind::Sidecar | MtpKind::DeepSeek => {
            req.mtp_mode != MtpMode::Off
                && (req.mtp_path.is_some() || facts.mtp_loaded)
                && draft >= caps.spec_draft_min
        }
        MtpKind::BoundOnly | MtpKind::None => false,
    };
    // Native skips the slab when DS4_SERVER_FORK_PARTIAL=0.
    let partial = quote_partial(req, caps, facts);
    // Qwen/Step allocate a complete graph per bank. Shared scratch would
    // let auto approve two banks when only one graph fits.
    // Example: Qwen MTP enable is another QSA+hidden per graph, not one
    // shared draft row.
    let (per_bank, scratch, mtp_state, checkpoint) = match (caps.family, shape) {
        (ModelFamily::Qwen4Exp, Some(s)) => {
            let graph = qwen_graph_bytes(s, ctx_tokens, native);
            let mtp = if mtp_on {
                qwen_mtp_enable_bytes(s, ctx_tokens, native)
            } else {
                0
            };
            let pool = if partial {
                qwen_checkpoint_pool_bytes(s)
            } else {
                0
            };
            (kv.saturating_add(graph).saturating_add(mtp), 0, 0, pool)
        }
        (ModelFamily::Step37, Some(s)) => {
            let graph = step_graph_bytes(s, native);
            let spec = if step_mtp {
                step_spec_bytes(s, ctx, native)
            } else {
                0
            };
            let pool = if partial {
                step_checkpoint_pool_bytes(s, step_mtp)
            } else {
                0
            };
            let logits = u64::from(s.n_vocab) * SIZEOF_F32;
            (
                kv.saturating_add(graph)
                    .saturating_add(spec)
                    .saturating_add(logits),
                0,
                0,
                pool,
            )
        }
        (ModelFamily::SolarOpen2, Some(s)) => {
            let scratch = family_graph_scratch(s, native);
            let pool = if partial {
                solar_checkpoint_pool_bytes(s)
            } else {
                0
            };
            (kv, scratch, 0, pool)
        }
        (ModelFamily::Motif3, Some(s)) => {
            let scratch = motif_graph_bytes(s, native);
            let pool = if partial {
                motif_checkpoint_pool_bytes(s)
            } else {
                0
            };
            let bank_outputs = (u64::from(s.n_embd) + u64::from(s.n_vocab)) * SIZEOF_F32;
            (kv.saturating_add(bank_outputs), scratch, 0, pool)
        }
        (ModelFamily::Inkling, Some(s)) => {
            let (base, with_mtp) = inkling_runtime_bytes(s, ctx, native);
            (if sidecar_loaded { with_mtp } else { base }, 0, 0, 0)
        }
        (ModelFamily::Dots3Note, Some(s)) => (dots3_graph_bytes(s, ctx, native), 0, 0, 0),
        (ModelFamily::Glm53, Some(s)) => (glm_graph_bytes(s, ctx), 0, 0, 0),
        (ModelFamily::DeepSeek4, Some(s)) => {
            // The shared graph owns its own caches before bank slabs are fitted.
            let mut scratch = deepseek_graph_bytes(s, ctx, native);
            let mut bank = deepseek_bank_bytes(s, ctx, native);
            let mtp = if sidecar_loaded {
                bank += deepseek_mtp_bank_bytes(s, ctx, native);
                deepseek_mtp_bytes(s, ctx, native)
            } else {
                0
            };
            // The successful sidecar probe is preserved across engine open.
            // Loading allocates DSpark state even with DS4_CONT_DSPARK=0.
            if facts.dspark_ok == Some(true) {
                let (shared, per_bank) = deepseek_dspark_bytes(s, ctx, native);
                scratch += shared;
                bank += per_bank;
            }
            if !quote_batch_alloc(req, caps, facts) {
                bank = 0;
            }
            // The slab price already includes every rollback checkpoint depth.
            (bank, scratch, mtp, 0)
        }
        (_, Some(s)) => {
            let scratch = family_graph_scratch(s, native);
            let pool = if partial { kv } else { 0 };
            let mtp_state = if mtp_on {
                u64::from(s.n_embd)
                    .saturating_mul(u64::from(s.n_layer))
                    .saturating_mul(caps.spec_draft_min.max(1) as u64)
                    .saturating_mul(2)
            } else {
                0
            };
            (kv, scratch, mtp_state, pool)
        }
        _ => (kv, 0, 0, 0),
    };
    let media = if caps.family == ModelFamily::Step37 {
        if host.vision || facts.vision_loaded {
            shape.map(|s| step_media_bytes(s, ctx)).unwrap_or(GIB)
        } else {
            0
        }
    } else if caps.media_serial || host.vision {
        shape
            .map(|s| u64::from(s.n_embd).saturating_mul(8192).saturating_mul(2))
            .unwrap_or(GIB)
    } else {
        0
    };

    facts.shared_weights_bytes = Some(host.weights_bytes.saturating_add(host.mtp_bytes));
    facts.per_bank_bytes = Some(per_bank);
    facts.mtp_state_bytes = Some(mtp_state);
    facts.scratch_bytes = Some(scratch);
    facts.checkpoint_pool_bytes = Some(checkpoint);
    facts.ple_bytes = Some(ple_cache_bytes(caps));
    facts.media_reserve_bytes = Some(media);
    facts.fit_headroom_bytes = Some(quote_fit_headroom(req, caps, facts));
    // Zero is an unsupported probe, not a host with no RAM.
    facts.host_available_bytes = (host.available_bytes > 0).then_some(host.available_bytes);
    facts.native_chunk = Some(native);
}

/// Live host observation + GGUF span. Used by the CLI pre-open and post-fit.
///
/// After `Model::open` / fit, mapped weights and the fitted runtime have
/// already left MemAvailable. `resident` credits both so the quote is not
/// charged twice.
pub fn attach_host_quote(
    facts: &mut EngineFacts,
    req: &ServingRequest,
    caps: ServingCaps,
    shape: Option<Shape>,
    model_path: Option<&Path>,
    mtp_path: Option<&Path>,
    vision_path: Option<&Path>,
    dspark_path: Option<&Path>,
    split_count: u32,
    vision: bool,
    resident: bool,
) {
    let mut weights_bytes = model_path
        .map(|path| gguf_span_bytes(path, split_count))
        .unwrap_or(0);
    let mut mtp_bytes = artifact_span_bytes(mtp_path);
    let vision_bytes = artifact_span_bytes(vision_path);
    let dspark_bytes = artifact_span_bytes(dspark_path);
    // Weight-server already holds imported spans; MemAvailable includes them.
    match ipc_weight_skip() {
        IpcSkip::None => {}
        IpcSkip::Base => {
            weights_bytes = 0;
        }
        IpcSkip::Mtp => {
            mtp_bytes = 0;
        }
        IpcSkip::Both => {
            weights_bytes = 0;
            mtp_bytes = 0;
        }
    }

    let mapped = weights_bytes
        .saturating_add(mtp_bytes)
        .saturating_add(vision_bytes)
        .saturating_add(dspark_bytes);
    let live = host_available_bytes(req.backend);
    let host = QuoteHost {
        weights_bytes: weights_bytes
            .saturating_add(vision_bytes)
            .saturating_add(dspark_bytes),
        mtp_bytes,
        available_bytes: if live == 0 {
            0
        } else {
            quote_available(live, mapped, resident)
        },
        native_chunk: req.native_chunk,
        vision,
    };
    fill_quote_facts(facts, req, caps, shape, host);
    if resident {
        credit_resident(facts, live, mapped);
    }
}

enum IpcSkip {
    None,
    Base,
    Mtp,
    Both,
}

fn ipc_weight_skip() -> IpcSkip {
    let Ok(manifest) = std::env::var(WEIGHT_IPC_MANIFEST_ENV) else {
        return IpcSkip::None;
    };
    if manifest.is_empty() {
        return IpcSkip::None;
    }
    match std::env::var(WEIGHT_IPC_SCOPE_ENV).ok().as_deref() {
        Some("base") => IpcSkip::Base,
        Some("mtp") => IpcSkip::Mtp,
        Some("both") | None | Some("") => IpcSkip::Both,
        _ => IpcSkip::None,
    }
}

fn quote_available(live: u64, mapped: u64, resident: bool) -> u64 {
    if resident {
        live.saturating_add(mapped)
    } else {
        live
    }
}

// Sidecar GGUF may split independently of the base; unread metadata is one file.
fn artifact_span_bytes(path: Option<&Path>) -> u64 {
    let Some(path) = path else {
        return 0;
    };
    match GgufFile::open(path) {
        Ok(g) => gguf_span_bytes(path, g.split_count()),
        Err(_) => file_len(Some(path)),
    }
}

pub fn gguf_span_bytes(path: &Path, split_count: u32) -> u64 {
    let count = split_count.max(1);
    if count == 1 {
        return file_len(Some(path));
    }
    let path_s = path.to_string_lossy();
    let mut total: u64 = 0;
    for i in 0..count {
        let shard: PathBuf = model_split_sibling_path(&path_s, i, count)
            .map(PathBuf::from)
            .unwrap_or_else(|| path.to_path_buf());
        total = total.saturating_add(file_len(Some(&shard)));
    }
    if total == 0 {
        file_len(Some(path))
    } else {
        total
    }
}

pub fn host_available_bytes(backend: Backend) -> u64 {
    let device = match backend {
        Backend::Cuda => nvidia_fb_probe(),
        Backend::Metal | Backend::Cpu => None,
    };
    quote_ceiling(meminfo_available(), meminfo_total(), device)
}

// Discrete CUDA FB is much smaller than host RAM. Unified hosts (GB10)
// report N/A or a size matching RAM — keep MemAvailable.
fn quote_ceiling(avail: u64, total: u64, device: Option<(u64, u64)>) -> u64 {
    let Some((dev_total, dev_free)) = device else {
        return avail;
    };
    if dev_total == 0 {
        return avail;
    }
    if total > 0 && dev_total.saturating_add(GIB) < total {
        return dev_free;
    }
    avail
}

fn nvidia_fb_probe() -> Option<(u64, u64)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_nvidia_csv(std::str::from_utf8(&out.stdout).ok()?)
}

fn parse_nvidia_csv(text: &str) -> Option<(u64, u64)> {
    let line = text.lines().next()?.trim();
    let mut parts = line.split(',');
    let total = parse_nvidia_mib(parts.next()?)?;
    let free = parse_nvidia_mib(parts.next()?)?;
    Some((total, free))
}

fn parse_nvidia_mib(raw: &str) -> Option<u64> {
    let t = raw.trim();
    if t.eq_ignore_ascii_case("[n/a]") || t.eq_ignore_ascii_case("n/a") {
        return None;
    }
    let mib: u64 = t.parse().ok()?;
    Some(mib.saturating_mul(MIB))
}

fn family_native_chunk(caps: ServingCaps, ctx: u32) -> u32 {
    let ctx = ctx.max(1);
    let cap = match caps.family {
        ModelFamily::Qwen4Exp => env_u32(
            QWEN_PREFILL_CHUNK_ENV,
            QWEN_NATIVE_DEFAULT,
            1,
            QWEN_NATIVE_MAX,
        ),
        ModelFamily::Step37 => env_u32(
            STEP_PREFILL_CHUNK_ENV,
            STEP_NATIVE_DEFAULT,
            1,
            STEP_NATIVE_MAX,
        ),
        ModelFamily::Inkling => env_u32(
            INKLING_PREFILL_CHUNK_ENV,
            INKLING_NATIVE_DEFAULT,
            1,
            INKLING_NATIVE_MAX,
        ),
        ModelFamily::Glm53 => GLM_NATIVE_DEFAULT,
        ModelFamily::ExaoneMoe => env_u32(
            EXAONE_PREFILL_CHUNK_ENV,
            if caps.variant == Variant::K2Horizon375B {
                K2_NATIVE_DEFAULT
            } else {
                EXAONE_NATIVE_DEFAULT
            },
            1,
            FAMILY_NATIVE_MAX,
        ),
        ModelFamily::Motif3 => env_u32(
            MOTIF_PREFILL_CHUNK_ENV,
            MOTIF_NATIVE_DEFAULT,
            1,
            MOTIF_NATIVE_MAX,
        ),
        ModelFamily::SolarOpen2 => metal_native_chunk(ctx, SOLAR_NATIVE_DEFAULT),
        ModelFamily::DeepSeek4 => metal_native_chunk(ctx, DEFAULT_SCHED_CHUNK),
        ModelFamily::Dots3Note => env_u32(
            DOTS3_PREFILL_CHUNK_ENV,
            DOTS3_NATIVE_DEFAULT,
            1,
            DOTS3_NATIVE_MAX,
        ),
    };
    cap.min(ctx)
}

fn metal_native_chunk(ctx: u32, default: u32) -> u32 {
    let fallback = ctx.min(default).max(1);
    let Ok(raw) = std::env::var(SOLAR_PREFILL_CHUNK_ENV) else {
        return fallback;
    };
    if raw.is_empty() {
        return fallback;
    }
    let Ok(parsed) = raw.parse::<i64>() else {
        return fallback;
    };
    // C: value <= 0 pins the session cap to ctx.
    if parsed <= 0 {
        return ctx.max(1);
    }
    (parsed as u32).min(ctx).max(1)
}

fn env_u32(name: &str, fallback: u32, min: u32, max: u32) -> u32 {
    let Ok(raw) = std::env::var(name) else {
        return fallback;
    };
    if raw.is_empty() {
        return fallback;
    }
    let Ok(parsed) = raw.parse::<u32>() else {
        return fallback;
    };
    if parsed < min || parsed > max {
        return fallback;
    }
    parsed
}

fn qwen_qsa_score_rows(capacity: u32) -> u32 {
    if std::env::var_os(QWEN_QSA_NO_FUSED_ENV).is_some() {
        return capacity;
    }
    capacity.min(QWEN_QSA_SCORE_ROWS)
}

// C `qwen4exp_graph_bytes_estimate`. Each continuous bank owns one copy.
fn qwen_graph_bytes(shape: Shape, ctx: u32, cap: u32) -> u64 {
    if ctx < 4 || cap == 0 || cap > ctx {
        return 0;
    }
    let p = u64::from(cap);
    let ctx = u64::from(ctx);
    let hidden = u64::from(shape.n_embd);
    let hc = u64::from(shape.n_hc);
    let width = hidden.saturating_mul(hc);
    let blocks = ctx / QWEN_GRAPH_RATIO;
    let selected_blocks = blocks.min(QWEN_GRAPH_SELECTED_BLOCKS_MAX);
    let selected_tokens = u64::from(shape.n_indexer_top_k).saturating_add(QWEN_GRAPH_RATIO - 1);
    let index_q =
        u64::from(shape.n_indexer_head).saturating_mul(u64::from(shape.n_indexer_head_dim));
    let index_qk = index_q.saturating_add(u64::from(shape.n_indexer_head_dim));
    let q = u64::from(shape.n_head).saturating_mul(u64::from(shape.n_head_dim));
    let kv = u64::from(shape.n_head_kv).saturating_mul(u64::from(shape.n_head_dim));
    let key_dim = QWEN_GDN_KEY_HEADS.saturating_mul(QWEN_GDN_HEAD);
    let value_dim = QWEN_GDN_VALUE_HEADS.saturating_mul(QWEN_GDN_HEAD);
    let conv_dim = key_dim.saturating_mul(2).saturating_add(value_dim);
    let qsa_layers = u64::from(shape.n_full_attn_count);
    let gdn_layers = u64::from(shape.n_layer).saturating_sub(qsa_layers);
    let score_rows = u64::from(qwen_qsa_score_rows(cap));

    let mut bytes = 0u64;
    bytes = bytes.saturating_add(p.saturating_mul(SIZEOF_I32));
    bytes = bytes.saturating_add(
        p.saturating_mul(
            hidden
                .saturating_add(width.saturating_mul(2))
                .saturating_add(hidden),
        )
        .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(
        2u64.saturating_mul(u64::from(shape.n_vocab))
            .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(
        p.saturating_mul(
            width
                .saturating_mul(2)
                .saturating_add(QWEN_GRAPH_LOWRANK)
                .saturating_add(hidden)
                .saturating_add(hc.saturating_mul(2)),
        )
        .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(p.saturating_mul(hidden).saturating_mul(SIZEOF_U16));
    bytes = bytes.saturating_add(
        p.saturating_mul(
            hidden
                .saturating_mul(2)
                .saturating_add(width.saturating_mul(6)),
        )
        .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(width.saturating_mul(9).saturating_mul(SIZEOF_F32));
    bytes = bytes.saturating_add(
        p.saturating_mul(1 + QWEN_PLE_HOST_ID_LANES)
            .saturating_mul(SIZEOF_U64),
    );
    bytes = bytes.saturating_add(
        p.saturating_mul(
            conv_dim
                .saturating_mul(2)
                .saturating_add(value_dim.saturating_mul(3))
                .saturating_add(QWEN_GDN_VALUE_HEADS.saturating_mul(4)),
        )
        .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(
        gdn_layers
            .saturating_mul(
                conv_dim.saturating_mul(4).saturating_add(
                    QWEN_GDN_VALUE_HEADS
                        .saturating_mul(QWEN_GDN_HEAD)
                        .saturating_mul(QWEN_GDN_HEAD),
                ),
            )
            .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(
        p.saturating_mul(
            index_qk
                .saturating_add(index_q)
                .saturating_add(q.saturating_mul(5))
                .saturating_add(kv.saturating_mul(2)),
        )
        .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(p.saturating_mul(blocks).saturating_mul(SIZEOF_F32));
    bytes = bytes.saturating_add(p.saturating_mul(selected_blocks).saturating_mul(SIZEOF_U32));
    bytes = bytes.saturating_add(p.saturating_mul(selected_tokens).saturating_mul(SIZEOF_I32));
    bytes = bytes.saturating_add(p.saturating_mul(SIZEOF_U32));
    bytes = bytes.saturating_add(
        score_rows
            .saturating_mul(u64::from(shape.n_head))
            .saturating_mul(selected_tokens)
            .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(
        qsa_layers
            .saturating_mul(
                ctx.saturating_mul(u64::from(shape.n_indexer_head_dim))
                    .saturating_add(blocks.saturating_mul(u64::from(shape.n_indexer_head_dim)))
                    .saturating_add(ctx.saturating_mul(kv).saturating_mul(2)),
            )
            .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(
        p.saturating_mul(u64::from(shape.n_expert))
            .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(
        p.saturating_mul(u64::from(shape.n_expert_used))
            .saturating_mul(2)
            .saturating_mul(SIZEOF_F32),
    );
    bytes = bytes.saturating_add(
        p.saturating_mul(u64::from(shape.n_expert_used))
            .saturating_mul(
                u64::from(shape.n_ff_exp)
                    .saturating_mul(3)
                    .saturating_add(hidden),
            )
            .saturating_mul(SIZEOF_F32),
    );
    bytes.saturating_add(
        p.saturating_mul(
            u64::from(shape.n_ff_shexp)
                .saturating_mul(3)
                .saturating_add(hidden),
        )
        .saturating_mul(SIZEOF_F32),
    )
}

// C `qwen4exp_qsa_state_alloc` for one MTP QSA layer (`mtp_qsa_state`).
fn qwen_qsa_state_bytes(shape: Shape, ctx: u32) -> u64 {
    let ctx = u64::from(ctx);
    let index = u64::from(shape.n_indexer_head_dim);
    let kv = u64::from(shape.n_head_kv).saturating_mul(u64::from(shape.n_head_dim));
    let blocks = ctx / QWEN_GRAPH_RATIO;
    ctx.saturating_mul(index)
        .saturating_add(blocks.saturating_mul(index))
        .saturating_add(ctx.saturating_mul(kv).saturating_mul(2))
        .saturating_mul(SIZEOF_F32)
}

// C `qwen4exp_graph_mtp_enable`: extra QSA + capacity hidden + pending HC.
fn qwen_mtp_enable_bytes(shape: Shape, ctx: u32, cap: u32) -> u64 {
    let width = u64::from(shape.n_embd).saturating_mul(u64::from(shape.n_hc));
    let hidden = u64::from(cap)
        .saturating_mul(width)
        .saturating_mul(SIZEOF_F32);
    let pending = width.saturating_mul(SIZEOF_F32);
    qwen_qsa_state_bytes(shape, ctx)
        .saturating_add(hidden)
        .saturating_add(pending)
}

fn qwen_checkpoint_slot_bytes(shape: Shape) -> u64 {
    let width = u64::from(shape.n_embd).saturating_mul(u64::from(shape.n_hc));
    let ple = width
        .saturating_mul(QWEN_PLE_CONV_TAPS)
        .saturating_mul(SIZEOF_F32);
    let key_dim = QWEN_GDN_KEY_HEADS.saturating_mul(QWEN_GDN_HEAD);
    let value_dim = QWEN_GDN_VALUE_HEADS.saturating_mul(QWEN_GDN_HEAD);
    let conv_dim = key_dim.saturating_mul(2).saturating_add(value_dim);
    let conv = conv_dim
        .saturating_mul(u64::from(shape.n_ssm_conv.max(1)))
        .saturating_mul(SIZEOF_F32);
    let recurrent = QWEN_GDN_VALUE_HEADS
        .saturating_mul(QWEN_GDN_HEAD)
        .saturating_mul(QWEN_GDN_HEAD)
        .saturating_mul(SIZEOF_F32);
    let gdn_layers = u64::from(shape.n_layer.saturating_sub(shape.n_full_attn_count));
    ple.saturating_add(gdn_layers.saturating_mul(conv.saturating_add(recurrent)))
}

fn qwen_checkpoint_pool_bytes(shape: Shape) -> u64 {
    qwen_checkpoint_slot_bytes(shape).saturating_mul(CHECKPOINT_SLOTS)
}

// C step37_ckpt_init: 32 slots of sliding-window KV, plus MTP windows/state.
fn step_checkpoint_slot_bytes(shape: Shape, mtp_on: bool) -> u64 {
    let period = shape.n_swa_period.max(1);
    let sliding = (0..shape.n_layer)
        .filter(|il| !il.is_multiple_of(period))
        .count() as u64;
    let window = u64::from(shape.n_swa.max(1));
    let row =
        2 * u64::from(shape.n_head_kv.max(1)) * u64::from(shape.n_head_dim.max(1)) * SIZEOF_U16;
    let mut slot = sliding.saturating_mul(window).saturating_mul(row);
    if mtp_on {
        let pred = u64::from(shape.n_nextn_predict.max(1));
        let state = u64::from(shape.n_embd).saturating_mul(SIZEOF_F32);
        slot = slot
            .saturating_add(pred.saturating_mul(window.saturating_mul(row).saturating_add(state)));
    }
    slot
}

fn step_checkpoint_pool_bytes(shape: Shape, mtp_on: bool) -> u64 {
    step_checkpoint_slot_bytes(shape, mtp_on).saturating_mul(CHECKPOINT_SLOTS)
}

// C `g->state_bytes`: KDA layers only (il % 4 != 0). 32 copies in the slab.
fn solar_checkpoint_slot_bytes(shape: Shape) -> u64 {
    let kda_dim = u64::from(shape.n_head).saturating_mul(u64::from(shape.n_kda_head_dim.max(1)));
    let recurrent = u64::from(shape.n_head)
        .saturating_mul(u64::from(shape.n_kda_head_dim.max(1)))
        .saturating_mul(u64::from(shape.n_kda_head_dim.max(1)))
        .saturating_mul(SIZEOF_F32);
    let conv = kda_dim
        .saturating_mul(u64::from(shape.n_ssm_conv.max(1)))
        .saturating_mul(SIZEOF_F32);
    let per = recurrent.saturating_add(conv.saturating_mul(3));
    let n_kda = (0..shape.n_layer).filter(|il| il % 4 != 0).count() as u64;
    per.saturating_mul(n_kda)
}

fn solar_checkpoint_pool_bytes(shape: Shape) -> u64 {
    solar_checkpoint_slot_bytes(shape).saturating_mul(CHECKPOINT_SLOTS)
}

// Native checkpoints contain only SWA windows; full prefixes stay in the bank.
fn motif_checkpoint_pool_bytes(shape: Shape) -> u64 {
    let sliding = (0..shape.n_layer)
        .filter(|&il| !motif_layer_is_full(shape, il))
        .count() as u64;
    sliding
        .saturating_mul(u64::from(shape.n_swa))
        .saturating_mul(motif_kv_row_bytes(shape))
        .saturating_mul(CHECKPOINT_SLOTS)
}

// Same gate as apply_env publishing DS4_SERVER_FORK_PARTIAL=1.
fn quote_partial(req: &ServingRequest, caps: ServingCaps, facts: &EngineFacts) -> bool {
    if caps.reuse != ReuseKind::Partial {
        return false;
    }
    match req.prefix_reuse {
        PrefixReuse::Off | PrefixReuse::Exact => false,
        PrefixReuse::Partial | PrefixReuse::Auto => {
            quote_bank_lane(req, caps, facts) && facts.partial_reuse != Some(false)
        }
    }
}

fn quote_bank_lane(req: &ServingRequest, caps: ServingCaps, facts: &EngineFacts) -> bool {
    req.lane != LaneMode::Serial
        && facts.cont_lane != Some(false)
        && quote_batch_alloc(req, caps, facts)
}

// Static coalescing still creates banks when the continuous driver is off.
fn quote_batch_alloc(req: &ServingRequest, caps: ServingCaps, facts: &EngineFacts) -> bool {
    if req.backend != Backend::Cuda || caps.banks == BankLane::Serial {
        return false;
    }

    let want = match req.max_seqs {
        MaxSeqs::Off => {
            return false;
        }
        MaxSeqs::Auto => match caps.banks {
            BankLane::Serial | BankLane::OptIn => 1,
            BankLane::Persistent => caps.qualified_banks.unwrap_or(DEFAULT_MAX_SEQS),
        },
        MaxSeqs::Fixed(n) => n,
    };
    let width = facts.banks_fitted.unwrap_or(want).min(want);

    caps.banks != BankLane::OptIn || width >= 2
}

// C metal_graph_alloc_bytes_estimate, including its initial cache set and
// 96 MiB allocator slack. Price both CUDA packed mirrors conservatively:
// native may refuse them when VMM is unavailable. F32 also bounds Metal F16.
fn deepseek_graph_bytes(s: Shape, ctx: u64, native: u32) -> u64 {
    let pc = u64::from(native);
    let dim = u64::from(s.n_head_dim);
    let index_dim = u64::from(s.n_indexer_head_dim);
    let hidden = u64::from(s.n_embd);
    let heads = u64::from(s.n_head);
    let hc = u64::from(s.n_hc);
    let groups = u64::from(s.n_out_group);
    let used = u64::from(s.n_expert_used);
    let ff = u64::from(s.n_ff_exp);
    let row = 4 * hc * hidden
        + 2 * (2 * hc + hc * hc)
        + 7 * hidden
        + 2 * u64::from(s.n_lora_q)
        + 2 * heads * dim
        + 2 * dim
        + 4 * dim.max(index_dim)
        + u64::from(s.n_indexer_head) * (index_dim + 1)
        + groups * u64::from(s.n_lora_o)
        + dim * (heads / groups)
        + u64::from(s.n_lora_o)
        + 3 * ff
        + 2 * u64::from(s.n_expert)
        + 3 * used * ff
        + used * hidden;
    let (cache, state) = deepseek_cache_bytes(s, ctx, native);
    let bytes = cache + 2 * state;
    bytes
        + (2 * (ctx / 4 + 2) * pc
            + u64::from(s.n_indexer_top_k) * pc
            + 129 * u64::from(s.n_vocab)
            + pc * row)
            * SIZEOF_F32
        + (96 << 20)
}

// Cache mirrors use the same conservative policy as the shared graph.
// Return cache bytes and one compressor state plane for slab-depth accounting.
fn deepseek_cache_bytes(s: Shape, ctx: u64, native: u32) -> (u64, u64) {
    let dim = u64::from(s.n_head_dim);
    let index_dim = u64::from(s.n_indexer_head_dim);
    let raw = deepseek_raw_cap(s, ctx, native);
    let mut cache = u64::from(s.n_layer) * raw * dim * SIZEOF_F32;
    let mut state = 0;
    let packed = dim - u64::from(s.n_rot)
        + u64::from(s.n_rot) * SIZEOF_F32
        + (dim - u64::from(s.n_rot)) / 64 * SIZEOF_F32;
    for il in 0..s.n_layer {
        let ratio = deepseek_comp_ratio(s, il);
        if ratio == 0 {
            continue;
        }
        let cap = ctx / ratio + 2;
        let coff = if ratio == 4 { 2 } else { 1 };
        cache += cap * (dim * SIZEOF_F32 + packed);
        state += coff * coff * dim * ratio * SIZEOF_F32;
        if ratio == 4 {
            cache += cap * (index_dim * SIZEOF_F32 + index_dim / 2 + index_dim / 32 * SIZEOF_F32);
            state += coff * coff * index_dim * ratio * SIZEOF_F32;
        }
    }
    (cache, state)
}

// C ds4_batch_slabs_bank_bytes, full-depth cache capacity (vmm_comp=false).
// Four rollback depths are always allocated, independent of MTP/reuse mode.
fn deepseek_bank_bytes(s: Shape, ctx: u64, native: u32) -> u64 {
    let (cache, state) = deepseek_cache_bytes(s, ctx, native);
    cache + (2 + 2 * 4) * state
}

fn deepseek_mtp_bank_bytes(s: Shape, ctx: u64, native: u32) -> u64 {
    let raw = deepseek_raw_cap(s, ctx, native) * u64::from(s.n_head_dim);
    let hc = u64::from(s.n_hc) * u64::from(s.n_embd);
    (raw + 7 * hc + 3 * u64::from(s.n_embd) + u64::from(s.n_vocab) + 3) * SIZEOF_F32
}

// C metal_graph_alloc_bytes_estimate(enable_dspark) + ds4_dspark_slabs_alloc.
fn deepseek_dspark_bytes(s: Shape, ctx: u64, native: u32) -> (u64, u64) {
    let raw = 3 * deepseek_raw_cap(s, ctx, native) * u64::from(s.n_head_dim);
    let workspace = u64::from(native) * (4 * u64::from(s.n_embd) + u64::from(s.n_head_dim));
    (
        (raw + workspace + 5 * u64::from(s.n_vocab)) * SIZEOF_F32,
        raw * SIZEOF_F32,
    )
}

fn deepseek_comp_ratio(s: Shape, il: u32) -> u64 {
    if il < 2 {
        if s.variant == Variant::Flash {
            0
        } else {
            128
        }
    } else if il.is_multiple_of(2) {
        4
    } else {
        128
    }
}

fn deepseek_raw_cap(s: Shape, ctx: u64, native: u32) -> u64 {
    let window = u64::from(s.n_swa).min(ctx).max(1);
    let default = (window + u64::from(native)).min(ctx).div_ceil(256) * 256;
    let raw = std::env::var("DS4_METAL_GRAPH_RAW_CAP")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default.min(8192));
    raw.min(8192).max(window).min(ctx)
}

// Loaded support graphs allocate rollback states even with speculation off.
fn deepseek_mtp_bytes(s: Shape, ctx: u64, native: u32) -> u64 {
    let dim = u64::from(s.n_head_dim);
    let mut floats = 2 * deepseek_raw_cap(s, ctx, native) * dim + 16 * u64::from(s.n_vocab);
    for il in 0..s.n_layer {
        let ratio = deepseek_comp_ratio(s, il);
        let coff = if ratio == 4 { 2 } else { 1 };
        let state_dim = dim
            + if ratio == 4 {
                u64::from(s.n_indexer_head_dim)
            } else {
                0
            };
        floats += 6 * coff * coff * state_dim * ratio;
    }
    floats * SIZEOF_F32
}

fn family_graph_scratch(shape: Shape, native: u32) -> u64 {
    u64::from(native)
        .saturating_mul(u64::from(shape.n_embd))
        .saturating_mul(u64::from(shape.n_layer))
        .saturating_mul(2)
}

// C step37_memory: GQA workspace plus Step controls, gates and RoPE tables.
fn step_graph_bytes(s: Shape, native: u32) -> u64 {
    let hidden = u64::from(s.n_embd);
    let heads = u64::from(s.n_head.max(s.n_swa_head));
    let head_dim = u64::from(s.n_head_dim);
    let kv = u64::from(s.n_head_kv) * head_dim;
    let used = u64::from(s.n_expert_used);
    let ff = u64::from(s.n_ff_exp);
    let common = 5 * hidden
        + 2 * heads * head_dim
        + 2 * kv
        + 3 * u64::from(s.n_ff_dense)
        + 3 * ff
        + u64::from(s.n_expert)
        + 2 * used
        + 3 * used * ff
        + used * hidden;
    let controls = 2 + heads + head_dim + head_dim / 2;
    (u64::from(native) * (common + controls) + head_dim * 3 / 4 + u64::from(s.n_vocab)) * SIZEOF_F32
}

// C step37_draft_bytes + step37_mtp_memory: predictor scratch/windows,
// combined embeddings, tail and joined verification input.
fn step_spec_bytes(s: Shape, ctx: u64, native: u32) -> u64 {
    let cap = u64::from(native);
    let pred = u64::from(s.n_nextn_predict);
    let rows = ctx.min(u64::from(s.n_swa) + cap);
    let row_bytes = 2 * u64::from(s.n_head_kv) * u64::from(s.n_head_dim) * SIZEOF_U16;
    step_graph_bytes(s, native)
        + pred * rows * row_bytes
        + (3 * cap + 2 * pred) * u64::from(s.n_embd) * SIZEOF_F32
}

// C step37_vision_bytes(728) plus the serial session's prepared image features.
fn step_media_bytes(s: Shape, ctx: u64) -> u64 {
    let grid = STEP_VISION_EDGE / STEP_VISION_PATCH;
    let hidden = u64::from(s.n_embd);
    let workspace = grid
        * grid
        * (3 * STEP_VISION_PATCH * STEP_VISION_PATCH
            + 10 * STEP_VISION_DIM
            + STEP_VISION_FFN
            + hidden / 16
            + 2)
        * SIZEOF_F32;
    workspace + ctx.min(STEP_MEDIA_ROWS) * hidden * SIZEOF_F32
}

// C inkling_context_memory / inkling_mtp_memory: return base and loaded-MTP
// totals. Local KV stays fixed at 512 rows, including for shorter contexts.
fn inkling_runtime_bytes(s: Shape, ctx: u64, native: u32) -> (u64, u64) {
    let hidden = u64::from(s.n_embd);
    let kv = u64::from(s.n_head_kv) * u64::from(s.n_head_dim);
    let layers = u64::from(s.n_layer);
    let globals = u64::from(s.n_full_attn_count);
    let local = u64::from(s.n_swa);
    let history = u64::from(s.n_ssm_conv.saturating_sub(1));
    let conv = 2 * kv + 2 * hidden;
    let kv_row = 2 * kv * SIZEOF_U16;
    let hidden_row = hidden * SIZEOF_F32;
    let used = u64::from(s.n_expert_used);
    let shared = u64::from(s.n_expert_shared);
    // Sum inkling_width: activations, relative attention, routing and FFN.
    let width = 7 * hidden
        + 4 * kv
        + u64::from(s.n_head) * (INKLING_REL_DIM + INKLING_GLOBAL_ROWS)
        + u64::from(s.n_expert)
        + 3 * shared
        + 2 * used
        + 3 * u64::from(s.n_ff_dense)
        + (used + shared) * hidden
        + 3 * shared * u64::from(s.n_ff_exp)
        + 2;
    let cap = u64::from(native);
    let raw = ((layers - globals) * local + globals * ctx) * kv_row
        + layers * history * conv * SIZEOF_F32;
    let scratch = (cap * width + u64::from(s.n_vocab)) * SIZEOF_F32;
    let pred = u64::from(s.n_nextn_predict);
    let verify = ctx.min(pred + 1);
    let draft_raw = ((pred - INKLING_DRAFT_GLOBALS) * local + INKLING_DRAFT_GLOBALS * ctx) * kv_row
        + pred * history * conv * SIZEOF_F32
        + pred * hidden_row;
    let journals = (layers + pred) * (verify * kv_row + (history + verify) * conv * SIZEOF_F32);
    let mtp = raw + draft_raw + 2 * scratch + (3 * cap + pred) * hidden_row + journals;
    (raw + scratch, mtp)
}

// C dots3_graph_memory_estimate, plus the decode partial buffer allocated by
// dots3_graph_alloc. The trailing bound-only MTP block owns no cache/norms.
fn dots3_graph_bytes(s: Shape, ctx: u64, native: u32) -> u64 {
    let hidden = u64::from(s.n_embd);
    let heads = u64::from(s.n_head);
    let latent = u64::from(s.n_kv_lora);
    let sliding_latent = u64::from(s.n_swa_kv_lora);
    let rot = u64::from(s.n_rot);
    let q_lora = u64::from(s.n_lora_q);
    let index = u64::from(s.n_indexer_head_dim);
    let index_heads = u64::from(s.n_indexer_head);
    let used = u64::from(s.n_expert_used);
    let ff = u64::from(s.n_ff_exp);
    let cap = u64::from(native);
    let mut cache = 0;
    let mut norms = 0;
    for il in 0..s.n_layer.saturating_sub(s.n_nextn_predict) {
        let full = il == 0 || (s.n_swa_period != 0 && il % s.n_swa_period == 1);
        let layer_latent = if full { latent } else { sliding_latent };
        let rows = if full {
            ctx
        } else {
            ctx.min(u64::from(s.n_swa) + cap)
        };
        cache += rows * (layer_latent + rot) * SIZEOF_U16;
        norms += q_lora + layer_latent + rot;
        if full {
            cache += rows * index * SIZEOF_F32;
            norms += 2 * index;
        }
    }
    let row_f32 = 6 * hidden
        + q_lora
        + heads * u64::from(s.n_key_mla)
        + 2 * heads * latent
        + 2 * sliding_latent
        + 2 * rot
        + heads * u64::from(s.n_value_mla)
        + heads
        + 3 * u64::from(s.n_ff_dense)
        + 3 * ff
        + u64::from(s.n_expert)
        + used
        + 3 * used * ff
        + used * hidden
        + index
        + index_heads * index
        + index_heads;
    let row_i32 = 2 + used + u64::from(s.n_indexer_top_k);
    let fixed = rot + hidden + u64::from(s.n_vocab) + norms;
    let partial = DOTS3_PARTIAL_ROWS * DOTS3_PARTIAL_SPLITS * heads * (sliding_latent + 4);
    cache
        + (cap * row_f32 + fixed + cap.min(DOTS3_INDEX_ROWS) * ctx + partial) * SIZEOF_F32
        + cap * row_i32 * SIZEOF_I32
}

// C glm53_graph_bytes_estimate: scalar workspace, KDA state/control tensors,
// and full DSA KV. The trailing prediction layer is not executed.
fn glm_graph_bytes(s: Shape, ctx: u64) -> u64 {
    let hidden = u64::from(s.n_embd);
    let hc_count = u64::from(s.n_hc);
    let hc = hc_count * hidden;
    let heads = u64::from(s.n_head);
    let qdim = heads * u64::from(s.n_key_mla);
    let kda_head = u64::from(s.n_kda_head_dim);
    let kda_dim = heads * kda_head;
    let used = u64::from(s.n_expert_used);
    let ff = u64::from(s.n_ff_dense).max(used * u64::from(s.n_ff_exp));
    let floats = 4 * hidden
        + 4 * hc
        + 3 * hc_count * hc_count
        + hc_count
        + 2 * u64::from(s.n_lora_q)
        + 2 * u64::from(s.n_kv_lora)
        + 4 * qdim
        + 3 * ff
        + used * hidden
        + used
        + u64::from(s.n_vocab);
    let workspace = (1 + used) * SIZEOF_I32 + floats * SIZEOF_F32;
    let conv = kda_dim * u64::from(s.n_ssm_conv) * SIZEOF_F32;
    let controls = 3 * conv + (heads + kda_dim + kda_head) * SIZEOF_F32;
    let state = kda_dim * kda_head * SIZEOF_F32 + 3 * conv;
    let exec = s.n_layer.saturating_sub(s.n_nextn_predict);
    let dsa = (0..exec)
        .filter(|il| il % GLM_ATTENTION_PERIOD == GLM_ATTENTION_PERIOD - 1)
        .count() as u64;
    workspace + (u64::from(exec) - dsa) * (state + controls) + dsa * ctx * qdim * 2 * SIZEOF_U16
}

// C motif3_graph_memory_estimate, excluding the separately quoted bank caches.
fn motif_graph_bytes(s: Shape, native: u32) -> u64 {
    let hidden = u64::from(s.n_embd);
    let hc = u64::from(s.n_hc);
    let heads = u64::from(s.n_head);
    let kv_heads = u64::from(s.n_head_kv);
    let head_dim = u64::from(s.n_head_dim);
    let value_dim = u64::from(s.n_value_dim);
    let clean_heads = heads - u64::from(s.n_noise_head);
    let latent = u64::from(s.n_kv_lora);
    let rot = u64::from(s.n_rot);
    let used = u64::from(s.n_expert_used);
    let ff = u64::from(s.n_ff_exp);
    let row_f32 = 3 * hc * hidden
        + 4 * hc
        + 2 * hc * hc
        + 9 * hidden
        + 2 * u64::from(s.n_lora_q)
        + 2 * heads * head_dim
        + 2 * clean_heads * value_dim
        + 3 * latent
        + rot
        + kv_heads * (head_dim - rot + value_dim)
        + 2 * heads * latent
        + kv_heads * (head_dim + value_dim)
        + clean_heads
        + 2 * heads * value_dim
        + 2 * heads
        + 3 * u64::from(s.n_ff_dense)
        + 3 * ff
        + 2 * u64::from(s.n_expert)
        + used
        + 3 * used * ff
        + used * hidden;
    let cap = u64::from(native);
    (cap * row_f32 + rot + u64::from(s.n_vocab)) * SIZEOF_F32 + cap * (2 + used) * SIZEOF_I32
}

// Preserve both bank-fit and serial-fallback margins. These are reserves,
// not resident memory: never credit them back after model open.
fn quote_fit_headroom(req: &ServingRequest, caps: ServingCaps, facts: &EngineFacts) -> u64 {
    let serial = quote_session_headroom(req);
    if !quote_batch_alloc(req, caps, facts)
        || !matches!(
            caps.family,
            ModelFamily::Motif3
                | ModelFamily::ExaoneMoe
                | ModelFamily::Step37
                | ModelFamily::DeepSeek4
        )
    {
        return serial;
    }
    if let Some(mb) = env_nonnegative_mb(FIT_HEADROOM_ENV) {
        return mb.saturating_mul(MIB).max(serial);
    }
    if std::env::var(FIT_DERIVED_ENV).as_deref() == Ok("0") {
        return (FIT_STATIC_MB * MIB).max(serial);
    }
    let burst = env_nonnegative_mb(FIT_BURST_ENV).unwrap_or(FIT_BURST_MB);
    req.mem_floor_gb
        .saturating_mul(GIB)
        .saturating_add(burst.saturating_mul(MIB))
        .max(serial)
}

fn quote_session_headroom(req: &ServingRequest) -> u64 {
    if req.backend != Backend::Cuda || std::env::var(SESSION_FIT_ENV).as_deref() == Ok("0") {
        return 0;
    }
    env_nonnegative_mb(SESSION_HEADROOM_ENV)
        .unwrap_or(SESSION_HEADROOM_MB)
        .saturating_mul(MIB)
}

// Native atol accepts a signed decimal prefix and maps nonnumeric text to 0.
fn env_nonnegative_mb(key: &str) -> Option<u64> {
    let raw = std::env::var(key).ok()?;
    if raw.is_empty() {
        return None;
    }
    let text = raw.trim_start();
    let negative = text.starts_with('-');
    let digits = text
        .strip_prefix('-')
        .or_else(|| text.strip_prefix('+'))
        .unwrap_or(text);
    let end = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    let n = digits[..end].parse::<u64>().unwrap_or(0);
    if negative && n != 0 {
        return None;
    }
    Some(n)
}

fn resident_runtime(facts: &EngineFacts) -> u64 {
    // A refused batch fit destroys its runtime. The serial graph is still
    // lazy, so only the model mappings may be credited on that fallback.
    if facts.cont_lane == Some(false) {
        return 0;
    }
    let banks = facts.banks_fitted.unwrap_or(1);
    facts
        .per_bank_bytes
        .unwrap_or(0)
        .saturating_mul(u64::from(banks))
        .saturating_add(facts.mtp_state_bytes.unwrap_or(0))
        .saturating_add(facts.scratch_bytes.unwrap_or(0))
        .saturating_add(facts.checkpoint_pool_bytes.unwrap_or(0))
        .saturating_add(facts.ple_bytes.unwrap_or(0))
        .saturating_add(facts.media_reserve_bytes.unwrap_or(0))
}

fn credit_resident(facts: &mut EngineFacts, live: u64, mapped: u64) {
    if live == 0 {
        return;
    }
    facts.host_available_bytes = Some(
        live.saturating_add(mapped)
            .saturating_add(resident_runtime(facts)),
    );
}

fn bank_kv_bytes(shape: Shape, ctx: u64, native: u32) -> u64 {
    if shape.family == ModelFamily::SolarOpen2 {
        let gqa = (0..shape.n_layer).filter(|il| il % 4 == 0).count() as u64;
        return gqa
            .saturating_mul(ctx)
            .saturating_mul(solar_kv_row_bytes(shape))
            .saturating_add(solar_checkpoint_slot_bytes(shape));
    }
    if shape.family == ModelFamily::Motif3 {
        // Native reserves an extra MTP window even when speculation is off.
        let sliding = ctx.min(u64::from(shape.n_swa) + 1 + u64::from(native));
        let rows = (0..=shape.n_layer).fold(0u64, |rows, il| {
            rows.saturating_add(if motif_layer_is_full(shape, il) {
                ctx
            } else {
                sliding
            })
        });
        return rows.saturating_mul(motif_kv_row_bytes(shape));
    }
    let row = if shape.n_kv_lora > 0 {
        u64::from(shape.n_kv_lora + shape.n_key_mla + shape.n_value_mla).max(1) * 2
    } else {
        2 * u64::from(shape.n_head_kv.max(1))
            * u64::from(shape.n_head_dim.max(shape.n_value_dim).max(1))
            * 2
    };
    let tokens = match shape.family {
        ModelFamily::ExaoneMoe => exaone_kv_tokens(shape, ctx, native),
        ModelFamily::Step37 => step_kv_tokens(shape, ctx, native),
        _ => u64::from(shape.n_layer).saturating_mul(ctx),
    };
    tokens.saturating_mul(row)
}

fn step_kv_tokens(shape: Shape, ctx: u64, native: u32) -> u64 {
    let sliding = ctx.min(u64::from(shape.n_swa) + u64::from(native));
    (0..shape.n_layer)
        .map(|il| {
            if il.is_multiple_of(shape.n_swa_period.max(1)) {
                ctx
            } else {
                sliding
            }
        })
        .sum()
}

fn motif_layer_is_full(shape: Shape, il: u32) -> bool {
    il < shape.n_layer && shape.n_swa_period != 0 && il.is_multiple_of(shape.n_swa_period)
}

fn motif_kv_row_bytes(shape: Shape) -> u64 {
    (u64::from(shape.n_kv_lora) + u64::from(shape.n_rot)) * SIZEOF_U16
}

// Match solar_kv_row_bytes, including the per-head quantization scales.
fn solar_kv_row_bytes(shape: Shape) -> u64 {
    let dim = u64::from(shape.n_head_kv) * u64::from(shape.n_head_dim);
    let scales = u64::from(shape.n_head_kv) * 2 * SIZEOF_U16;
    let format = std::env::var(SOLAR_KV_FORMAT_ENV).unwrap_or_default();
    match format.as_str() {
        "" | "hybrid" | "kfp8-vfp4" | "k-fp8/v-fp4" => dim + dim / 2 + scales,
        "fp8" | "e4m3" => dim * 2 + scales,
        "fp4" | "e2m1" => dim + scales,
        // Unknown values fail native open; keep their quote conservative.
        _ => dim * 2 * SIZEOF_U16,
    }
}

// C `exaone_graph_layer_kv_cap`: 12 LLLG global layers own ctx; the other
// 36 keep the 128-token window plus one prefill chunk.
fn exaone_kv_tokens(shape: Shape, ctx: u64, native: u32) -> u64 {
    let ctx_u32 = ctx.min(u64::from(u32::MAX)) as u32;
    let n_exec = shape.n_layer.saturating_sub(shape.n_nextn_predict);
    let prefill = native.min(ctx_u32);

    (0..n_exec)
        .map(|il| u64::from(exaone_layer_kv_cap(il, ctx_u32, prefill, shape)))
        .sum()
}

fn exaone_layer_kv_cap(il: u32, ctx: u32, prefill: u32, shape: Shape) -> u32 {
    if !exaone_layer_is_sliding(il, shape) {
        return ctx;
    }
    (u64::from(shape.n_swa) + u64::from(prefill)).min(u64::from(ctx)) as u32
}

fn exaone_layer_is_sliding(il: u32, shape: Shape) -> bool {
    shape.n_swa != 0
        && shape.n_swa_period != 0
        && (il % shape.n_swa_period) != shape.n_swa_period - 1
}

fn ple_cache_bytes(caps: ServingCaps) -> u64 {
    if caps.family != ModelFamily::Qwen4Exp {
        return 0;
    }
    ple_cache_mb().saturating_mul(MIB)
}

// C qwen4exp_engine_open_ple: only 512/1024/2048, else 2048.
fn ple_cache_mb() -> u64 {
    let Ok(raw) = std::env::var(PLE_CACHE_MB_ENV) else {
        return DEFAULT_PLE_CACHE_MB;
    };
    let Ok(mb) = raw.parse::<u64>() else {
        return DEFAULT_PLE_CACHE_MB;
    };
    if !ple_cache_mb_valid(mb) {
        return DEFAULT_PLE_CACHE_MB;
    }
    mb
}

fn ple_cache_mb_valid(mb: u64) -> bool {
    mb == PLE_CACHE_MB_512 || mb == PLE_CACHE_MB_1024 || mb == DEFAULT_PLE_CACHE_MB
}

fn file_len(path: Option<&Path>) -> u64 {
    path.and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0)
}

fn meminfo_total() -> u64 {
    meminfo_kb("MemTotal:")
}

fn meminfo_available() -> u64 {
    meminfo_kb("MemAvailable:")
}

fn meminfo_kb(prefix: &str) -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(prefix) else {
            continue;
        };
        let kb: u64 = rest
            .split_whitespace()
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(0);
        return kb.saturating_mul(1024);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::{
        resolve_plan, serving_caps, MaxSeqs, MtpMode, PrefixReuse, ReuseKind, PREFILL_CHUNK_FENCE,
    };
    use crate::shape::{
        Variant, SHAPE_DOTS3_NOTE_PREV, SHAPE_GLM53_FLASH, SHAPE_INKLING_SMALL,
        SHAPE_K2_HORIZON_375B, SHAPE_KEXAONE_236B, SHAPE_MOTIF3, SHAPE_QWEN38_FLASH_NEXT,
        SHAPE_SOLAR_OPEN2_250B, SHAPE_STEP37_FLASH,
    };
    use std::io::Write;

    #[test]
    fn fill_quote_facts_names_every_budget() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut facts = EngineFacts::default();
        let req = ServingRequest::default();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        fill_quote_facts(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            QuoteHost {
                weights_bytes: 10 * GIB,
                mtp_bytes: GIB,
                available_bytes: 100 * GIB,
                native_chunk: None,
                vision: false,
            },
        );
        assert_eq!(facts.shared_weights_bytes, Some(11 * GIB));
        assert!(facts.per_bank_bytes.unwrap() > 0);
        assert_eq!(facts.mtp_state_bytes, Some(0));
        assert_eq!(facts.scratch_bytes, Some(0));
        assert!(facts.checkpoint_pool_bytes.unwrap() > 0);
        assert!(facts.ple_bytes.unwrap() > 0);
        assert_eq!(facts.media_reserve_bytes, Some(0));
        assert_eq!(facts.host_available_bytes, Some(100 * GIB));
        assert_eq!(facts.native_chunk, Some(QWEN_NATIVE_DEFAULT));

        let plan = resolve_plan(&req, Some(caps), &facts);
        let quote = plan.quote.expect("host adapter must produce a quote");
        assert_eq!(quote.shared_weights, 11 * GIB);
        assert_eq!(quote.available, 100 * GIB);
        assert!(quote.per_bank > 0);
        assert_eq!(quote.mtp_state, 0);
        assert_eq!(quote.scratch, 0);
        assert!(quote.checkpoint_pool > 0);
        assert!(quote.ple > 0);
        assert_eq!(quote.floor, req.mem_floor_gb * GIB);
        assert!(!plan.to_json()["quote"].is_null());
    }

    #[test]
    fn qwen_invalid_ple_mb_falls_back() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let req = ServingRequest::default();
        {
            let _ple = EnvGuard::set(PLE_CACHE_MB_ENV, "512");
            let facts = fill_qwen(&req, qwen_host(None));
            assert_eq!(facts.ple_bytes, Some(PLE_CACHE_MB_512 * MIB));
        }
        for mb in ["0", "1", "768"] {
            let _ple = EnvGuard::set(PLE_CACHE_MB_ENV, mb);
            let facts = fill_qwen(&req, qwen_host(None));
            assert_eq!(
                facts.ple_bytes,
                Some(DEFAULT_PLE_CACHE_MB * MIB),
                "env {mb}"
            );
        }
    }

    #[test]
    fn attach_host_quote_reads_the_mapped_span() {
        let _env = lock_test_env();
        let _man = EnvGuard::unset(WEIGHT_IPC_MANIFEST_ENV);
        let _scope = EnvGuard::unset(WEIGHT_IPC_SCOPE_ENV);
        let dir = std::env::temp_dir().join(format!("ds4-quote-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let a = dir.join("model-00001-of-00002.gguf");
        let b = dir.join("model-00002-of-00002.gguf");
        std::fs::File::create(&a)
            .unwrap()
            .write_all(&[0u8; 100])
            .unwrap();
        std::fs::File::create(&b)
            .unwrap()
            .write_all(&[0u8; 40])
            .unwrap();

        assert_eq!(gguf_span_bytes(&a, 2), 140);

        let mut facts = EngineFacts::default();
        let req = ServingRequest::default();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        attach_host_quote(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            Some(&a),
            None,
            None,
            None,
            2,
            false,
            false,
        );
        assert_eq!(facts.shared_weights_bytes, Some(140));
        assert!(facts.host_available_bytes.unwrap() > 0);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(plan.quote.is_some(), "{:?}", plan.to_json());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_host_probe_does_not_quote() {
        let _env = lock_test_env();
        let mut facts = EngineFacts::default();
        let req = ServingRequest::default();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        fill_quote_facts(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            QuoteHost {
                weights_bytes: 10 * GIB,
                mtp_bytes: GIB,
                available_bytes: 0,
                native_chunk: None,
                vision: false,
            },
        );
        assert_eq!(facts.host_available_bytes, None);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(plan.quote.is_none(), "{:?}", plan.to_json());
        assert!(
            !plan.issues.iter().any(|i| i.code == "quote_overflow"),
            "{:?}",
            plan.issues
        );
    }

    #[test]
    fn quote_includes_sidecar_spans() {
        let _env = lock_test_env();
        let _man = EnvGuard::unset(WEIGHT_IPC_MANIFEST_ENV);
        let _scope = EnvGuard::unset(WEIGHT_IPC_SCOPE_ENV);
        let dir = std::env::temp_dir().join(format!("ds4-quote-sidecars-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let a = dir.join("model-00001-of-00002.gguf");
        let b = dir.join("model-00002-of-00002.gguf");
        let mtp = dir.join("mtp.gguf");
        let vision = dir.join("vision.gguf");
        let dspark = dir.join("dspark.gguf");
        std::fs::File::create(&a)
            .unwrap()
            .write_all(&[0u8; 100])
            .unwrap();
        std::fs::File::create(&b)
            .unwrap()
            .write_all(&[0u8; 40])
            .unwrap();
        std::fs::File::create(&mtp)
            .unwrap()
            .write_all(&[0u8; 25])
            .unwrap();
        std::fs::File::create(&vision)
            .unwrap()
            .write_all(&[0u8; 17])
            .unwrap();
        std::fs::File::create(&dspark)
            .unwrap()
            .write_all(&[0u8; 11])
            .unwrap();

        {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "mtp");
            let facts = attach_ipc(&a, Some(&mtp), Some(&vision), Some(&dspark), false);
            assert_eq!(facts.shared_weights_bytes, Some(168));
        }

        let facts = attach_ipc(&a, Some(&mtp), Some(&vision), Some(&dspark), false);
        assert_eq!(facts.shared_weights_bytes, Some(193));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_available_bytes_reads_this_host() {
        let _env = lock_test_env();
        assert!(host_available_bytes(Backend::Cuda) > 0);
    }

    #[test]
    fn resident_span_is_credited_to_available() {
        let _env = lock_test_env();
        assert_eq!(quote_available(30 * GIB, 80 * GIB, false), 30 * GIB);
        assert_eq!(quote_available(30 * GIB, 80 * GIB, true), 110 * GIB);

        let mut cold = EngineFacts::default();
        let mut hot = EngineFacts::default();
        let req = ServingRequest::default();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let leftover = 30 * GIB;
        let mapped = 80 * GIB;
        fill_quote_facts(
            &mut cold,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            QuoteHost {
                weights_bytes: mapped,
                mtp_bytes: 0,
                available_bytes: leftover,
                native_chunk: None,
                vision: false,
            },
        );
        fill_quote_facts(
            &mut hot,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            QuoteHost {
                weights_bytes: mapped,
                mtp_bytes: 0,
                available_bytes: quote_available(leftover, mapped, true),
                native_chunk: None,
                vision: false,
            },
        );
        let cold_plan = resolve_plan(&req, Some(caps), &cold);
        let hot_plan = resolve_plan(&req, Some(caps), &hot);
        assert!(
            cold_plan.has_errors(),
            "leftover without credit must overflow"
        );
        assert!(
            cold_plan.issues.iter().any(|i| i.code == "quote_overflow"),
            "{:?}",
            cold_plan.issues
        );
        assert!(!hot_plan.has_errors(), "{:?}", hot_plan.issues);
        assert_eq!(hot_plan.effective.max_seqs, 2);
    }

    // catalog-parity runs `cargo test -p ds4-core` without --test-threads=1.
    // Quote helpers read process env, so those tests must not overlap.
    fn lock_test_env() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev }
        }

        fn unset(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prev.as_ref() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn qwen_host(native_chunk: Option<u32>) -> QuoteHost {
        QuoteHost {
            weights_bytes: 10 * GIB,
            mtp_bytes: 0,
            available_bytes: 100 * GIB,
            native_chunk,
            vision: false,
        }
    }

    fn fill_qwen(req: &ServingRequest, host: QuoteHost) -> EngineFacts {
        let mut facts = EngineFacts::default();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        fill_quote_facts(&mut facts, req, caps, Some(SHAPE_QWEN38_FLASH_NEXT), host);
        facts
    }

    #[test]
    fn qwen_native_defaults_to_runtime_256() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let req = ServingRequest::default();
        let facts = fill_qwen(&req, qwen_host(None));
        let native = QWEN_NATIVE_DEFAULT.min(req.ctx.max(1) as u32);
        assert_eq!(facts.native_chunk, Some(native));
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert_eq!(plan.effective.native_chunk, Some(native));
        assert_ne!(plan.effective.native_chunk, Some(PREFILL_CHUNK_FENCE));
    }

    #[test]
    fn qwen_native_reads_prefill_chunk_env() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(QWEN_PREFILL_CHUNK_ENV, "512");
        let req = ServingRequest::default();
        let facts = fill_qwen(&req, qwen_host(None));
        assert_eq!(facts.native_chunk, Some(512));
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert_eq!(plan.effective.native_chunk, Some(512));
    }

    #[test]
    fn qwen_native_chunk_does_not_raise_past_c() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut req = ServingRequest::default();
        req.native_chunk = Some(8192);
        let facts = fill_qwen(&req, qwen_host(Some(8192)));
        assert_eq!(facts.native_chunk, Some(QWEN_NATIVE_DEFAULT));
    }

    #[test]
    fn qwen_explicit_yield_past_runtime_native_errors() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut req = ServingRequest::default();
        req.sched_chunk = Some(512);
        let facts = fill_qwen(&req, qwen_host(None));
        assert_eq!(facts.native_chunk, Some(QWEN_NATIVE_DEFAULT));
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(plan.has_errors(), "{:?}", plan.issues);
        assert!(
            plan.issues.iter().any(|i| i.code == "chunk_past_native"),
            "{:?}",
            plan.issues
        );
        assert!(!plan.may_listen());
    }

    fn facts_cost(facts: &EngineFacts, req: &ServingRequest, banks: u32) -> u64 {
        facts
            .shared_weights_bytes
            .unwrap_or(0)
            .saturating_add(
                facts
                    .per_bank_bytes
                    .unwrap_or(0)
                    .saturating_mul(u64::from(banks)),
            )
            .saturating_add(facts.mtp_state_bytes.unwrap_or(0))
            .saturating_add(facts.scratch_bytes.unwrap_or(0))
            .saturating_add(facts.checkpoint_pool_bytes.unwrap_or(0))
            .saturating_add(facts.ple_bytes.unwrap_or(0))
            .saturating_add(facts.media_reserve_bytes.unwrap_or(0))
            .saturating_add(
                req.mem_floor_gb
                    .saturating_mul(GIB)
                    .max(facts.fit_headroom_bytes.unwrap_or(0)),
            )
    }

    #[test]
    fn qwen_two_bank_quote_charges_each_graph() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(QWEN_PREFILL_CHUNK_ENV, "8192");
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 0;
        let facts = fill_qwen(&req, qwen_host(None));
        let kv = bank_kv_bytes(SHAPE_QWEN38_FLASH_NEXT, req.ctx.max(1) as u64, 0);
        let per_bank = facts.per_bank_bytes.unwrap();
        assert_eq!(facts.scratch_bytes, Some(0));
        assert_eq!(
            facts.checkpoint_pool_bytes,
            Some(qwen_checkpoint_pool_bytes(SHAPE_QWEN38_FLASH_NEXT))
        );
        assert!(per_bank > kv, "each bank owns a graph, not only KV");
        assert_eq!(
            facts_cost(&facts, &req, 2) - facts_cost(&facts, &req, 1),
            per_bank
        );

        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let plan = resolve_plan(&req, Some(caps), &facts);
        let quote = plan.quote.expect("quote");
        assert_eq!(quote.scratch, 0);
        assert_eq!(quote.per_bank, per_bank);
        assert_eq!(quote.total, facts_cost(&facts, &req, quote.banks));
    }

    #[test]
    fn qwen_tight_budget_does_not_quote_two_graphs() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(QWEN_PREFILL_CHUNK_ENV, "8192");
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 0;
        let sized = fill_qwen(&req, qwen_host(None));
        let kv = bank_kv_bytes(SHAPE_QWEN38_FLASH_NEXT, req.ctx.max(1) as u64, 0);
        let per_bank = sized.per_bank_bytes.unwrap();
        assert!(per_bank > kv);
        let cost1 = facts_cost(&sized, &req, 1);
        let cost2 = facts_cost(&sized, &req, 2);
        assert!(
            cost2 > cost1 + kv,
            "second bank must add a graph, not only KV"
        );

        let mut host = qwen_host(None);
        host.available_bytes = cost1 + (per_bank - kv) / 2;
        assert!(host.available_bytes >= cost1);
        assert!(host.available_bytes < cost2);
        let facts = fill_qwen(&req, host);
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert_eq!(plan.effective.max_seqs, 1, "{:?}", plan.to_json());
        assert!(
            !plan.issues.iter().any(|i| i.code == "quote_overflow"),
            "{:?}",
            plan.issues
        );
    }

    #[test]
    fn qwen_mtp_is_charged_per_bank() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut off_req = ServingRequest::default();
        off_req.mtp_mode = MtpMode::Off;
        off_req.mem_floor_gb = 0;
        let mut on_req = off_req.clone();
        on_req.mtp_mode = MtpMode::On;
        let off = fill_qwen(&off_req, qwen_host(None));
        let on = fill_qwen(&on_req, qwen_host(None));
        let mtp = on.per_bank_bytes.unwrap() - off.per_bank_bytes.unwrap();
        assert!(
            mtp > 8 * MIB,
            "MTP QSA+hidden is per bank, not a shared draft row"
        );
        assert_eq!(on.mtp_state_bytes, Some(0));
        assert_eq!(off.mtp_state_bytes, Some(0));
        assert_eq!(
            facts_cost(&on, &on_req, 2) - facts_cost(&off, &off_req, 2),
            2 * mtp
        );
    }

    #[test]
    fn qwen_auto_draft_1_skips_mtp_enable() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut off = ServingRequest::default();
        off.mtp_mode = MtpMode::Off;
        off.mem_floor_gb = 0;
        let mut draft1 = off.clone();
        draft1.mtp_mode = MtpMode::Auto;
        draft1.mtp_draft = Some(1);
        let mut draft2 = draft1.clone();
        draft2.mtp_draft = Some(2);
        let off_facts = fill_qwen(&off, qwen_host(None));
        let d1 = fill_qwen(&draft1, qwen_host(None));
        let d2 = fill_qwen(&draft2, qwen_host(None));
        assert_eq!(d1.per_bank_bytes, off_facts.per_bank_bytes);
        assert!(
            d2.per_bank_bytes.unwrap() > d1.per_bank_bytes.unwrap(),
            "draft 2 must charge per-bank MTP enable"
        );
    }

    #[test]
    fn qwen_checkpoint_uses_recurrent_slots() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 0;
        let facts = fill_qwen(&req, qwen_host(None));
        let kv = bank_kv_bytes(SHAPE_QWEN38_FLASH_NEXT, req.ctx.max(1) as u64, 0);
        let pool = facts.checkpoint_pool_bytes.unwrap();
        assert_ne!(pool, kv, "checkpoint slab is recurrent slots, not one KV");
        assert_eq!(pool % CHECKPOINT_SLOTS, 0);
        assert!(pool > 2 * GIB, "32 GDN/PLE slots are several GiB");
    }

    #[test]
    fn qwen_off_reuse_skips_checkpoint_pool() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 0;
        req.prefix_reuse = PrefixReuse::Off;
        let facts = fill_qwen(&req, qwen_host(None));
        assert_eq!(facts.checkpoint_pool_bytes, Some(0));
        req.prefix_reuse = PrefixReuse::Exact;
        let facts = fill_qwen(&req, qwen_host(None));
        assert_eq!(facts.checkpoint_pool_bytes, Some(0));
    }

    #[test]
    fn qwen_auto_reuse_follows_resolved_lane() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 0;
        req.prefix_reuse = PrefixReuse::Auto;
        req.max_seqs = MaxSeqs::Off;
        let facts = fill_qwen(&req, qwen_host(None));
        assert_eq!(facts.checkpoint_pool_bytes, Some(0));
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert_eq!(plan.effective.prefix_reuse, ReuseKind::Exact);
        assert!(plan
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_SERVER_FORK_PARTIAL" && v == "0"));

        req.max_seqs = MaxSeqs::Fixed(2);
        let facts = fill_qwen(&req, qwen_host(None));
        assert_eq!(
            facts.checkpoint_pool_bytes,
            Some(qwen_checkpoint_pool_bytes(SHAPE_QWEN38_FLASH_NEXT))
        );
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert_eq!(plan.effective.prefix_reuse, ReuseKind::Partial);
        assert!(plan
            .env_overrides()
            .iter()
            .any(|(k, v)| k == "DS4_SERVER_FORK_PARTIAL" && v == "1"));
    }

    #[test]
    fn failed_fit_credits_only_model() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let req = ServingRequest {
            mem_floor_gb: 4,
            ..ServingRequest::default()
        };
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let mapped = 10 * GIB;
        for (live, can_listen) in [(4 * GIB, false), (40 * GIB, true)] {
            // Same facts as the server's batch_ctx_fit Err fallback.
            let mut facts = EngineFacts {
                banks_fitted: Some(1),
                cont_lane: Some(false),
                partial_reuse: Some(false),
                ..EngineFacts::default()
            };
            fill_quote_facts(
                &mut facts,
                &req,
                caps,
                Some(SHAPE_QWEN38_FLASH_NEXT),
                QuoteHost {
                    weights_bytes: mapped,
                    available_bytes: live,
                    ..qwen_host(None)
                },
            );
            credit_resident(&mut facts, live, mapped);
            let plan = resolve_plan(&req, Some(caps), &facts);
            assert_eq!(plan.may_listen(), can_listen, "{:?}", plan.issues);
            assert_eq!(facts.host_available_bytes, Some(live + mapped));
            assert_eq!(resident_runtime(&facts), 0);
            assert_eq!(
                plan.issues.iter().any(|i| i.code == "quote_overflow"),
                !can_listen
            );
        }
    }

    #[test]
    fn fitted_runtime_is_credited_after_fit() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 4;
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let mut facts = EngineFacts {
            banks_fitted: Some(2),
            ..EngineFacts::default()
        };
        let leftover = 4 * GIB;
        let mapped = 10 * GIB;
        fill_quote_facts(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            QuoteHost {
                weights_bytes: mapped,
                mtp_bytes: 0,
                available_bytes: leftover,
                native_chunk: None,
                vision: false,
            },
        );
        credit_resident(&mut facts, leftover, mapped);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(
            !plan.issues.iter().any(|i| i.code == "quote_overflow"),
            "{:?}",
            plan.issues
        );
        assert!(plan.may_listen(), "{:?}", plan.issues);
        assert_eq!(plan.effective.max_seqs, 2, "{:?}", plan.to_json());
    }

    #[test]
    fn step_graph_is_charged_per_bank() {
        let _env = lock_test_env();
        let mut req = ServingRequest::default();
        req.mtp_mode = MtpMode::Off;
        req.mem_floor_gb = 0;
        let mut facts = EngineFacts::default();
        let caps = serving_caps(ModelFamily::Step37, Variant::Step37Flash);
        fill_quote_facts(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_STEP37_FLASH),
            QuoteHost {
                weights_bytes: 10 * GIB,
                mtp_bytes: 0,
                available_bytes: 100 * GIB,
                native_chunk: None,
                vision: false,
            },
        );
        let kv = bank_kv_bytes(SHAPE_STEP37_FLASH, req.ctx.max(1) as u64, 0);
        let per_bank = facts.per_bank_bytes.unwrap();
        assert_eq!(facts.scratch_bytes, Some(0));
        assert!(per_bank > kv, "each Step bank owns a graph");
        assert_eq!(
            facts_cost(&facts, &req, 2) - facts_cost(&facts, &req, 1),
            per_bank
        );
    }

    #[test]
    fn step_checkpoint_uses_sliding_slots() {
        let _env = lock_test_env();
        let mut req = ServingRequest::default();
        req.mtp_mode = MtpMode::Off;
        req.mem_floor_gb = 0;
        req.max_seqs = MaxSeqs::Fixed(2);
        req.prefix_reuse = PrefixReuse::Partial;
        let facts = fill_family(
            ModelFamily::Step37,
            Variant::Step37Flash,
            SHAPE_STEP37_FLASH,
            &req,
            QuoteHost {
                weights_bytes: 10 * GIB,
                mtp_bytes: 0,
                available_bytes: 100 * GIB,
                native_chunk: None,
                vision: false,
            },
        );
        let kv = bank_kv_bytes(SHAPE_STEP37_FLASH, req.ctx.max(1) as u64, 0);
        let pool = facts.checkpoint_pool_bytes.unwrap();
        // C step37_ckpt_init: 32 slots of every sliding layer's 512-row
        // window × 2 × kv × 2 bytes. Full-attn layers stay in the bank.
        let sliding = (0..SHAPE_STEP37_FLASH.n_layer)
            .filter(|il| !il.is_multiple_of(SHAPE_STEP37_FLASH.n_swa_period))
            .count() as u64;
        let row = 2
            * u64::from(SHAPE_STEP37_FLASH.n_head_kv)
            * u64::from(SHAPE_STEP37_FLASH.n_head_dim)
            * SIZEOF_U16;
        let want = sliding
            .saturating_mul(u64::from(SHAPE_STEP37_FLASH.n_swa))
            .saturating_mul(row)
            .saturating_mul(CHECKPOINT_SLOTS);
        assert_eq!(pool, want);
        assert!(pool > kv, "32 SWA slots exceed one full-context KV");
    }

    #[test]
    fn motif_graph_matches_native() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(MOTIF_PREFILL_CHUNK_ENV, "4096");
        let mut req = ServingRequest::default();
        req.ctx = 262144;
        let facts = fill_family(
            ModelFamily::Motif3,
            Variant::Motif3,
            SHAPE_MOTIF3,
            &req,
            qwen_host(None),
        );
        // Sum of unconditional M3_ALLOC requests, without bank caches.
        assert_eq!(facts.scratch_bytes, Some(5_794_820_352));
        let mut tight = facts.clone();
        tight.host_available_bytes = Some(facts_cost(&facts, &req, 1) - GIB);
        assert!(resolve_plan(
            &req,
            Some(serving_caps(ModelFamily::Motif3, Variant::Motif3)),
            &tight
        )
        .has_errors());
    }

    #[test]
    fn motif_fit_margin_quote() {
        let _env = lock_test_env();
        let _headroom = EnvGuard::unset("DS4_BATCH_FIT_HEADROOM_MB");
        let _derived = EnvGuard::unset("DS4_BATCH_FIT_HEADROOM_DERIVED");
        let _burst = EnvGuard::unset("DS4_BATCH_FIT_BURST_MB");
        let caps = serving_caps(ModelFamily::Motif3, Variant::Motif3);
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 4;
        req.ctx = 262144;
        for (key, value, expected) in [
            ("DS4_BATCH_FIT_BURST_MB", "2048", 6 * GIB),
            ("DS4_BATCH_FIT_BURST_MB", "3072", 7 * GIB),
            ("DS4_BATCH_FIT_HEADROOM_DERIVED", "0", 6 * GIB),
            ("DS4_BATCH_FIT_HEADROOM_MB", "8192", 8 * GIB),
            ("DS4_BATCH_FIT_HEADROOM_MB", "1024", 4 * GIB),
        ] {
            let _setting = EnvGuard::set(key, value);
            let mut facts = fill_family(
                ModelFamily::Motif3,
                Variant::Motif3,
                SHAPE_MOTIF3,
                &req,
                qwen_host(None),
            );
            let quote = resolve_plan(&req, Some(caps), &facts).quote.unwrap();
            assert_eq!(quote.floor, expected, "{key}={value}");
            facts.host_available_bytes = Some(
                quote.total - quote.per_bank * u64::from(quote.banks) + 2 * quote.per_bank - 1,
            );
            let plan = resolve_plan(&req, Some(caps), &facts);
            assert_eq!(plan.effective.max_seqs, 1);
            assert!(!plan.has_errors(), "{:?}", plan.issues);
            req.max_seqs = MaxSeqs::Fixed(2);
            assert!(resolve_plan(&req, Some(caps), &facts).has_errors());
            req.max_seqs = MaxSeqs::Auto;
        }
    }

    #[test]
    fn solar_bank_uses_gqa_and_kda() {
        let _env = lock_test_env();
        // Native Solar: 12 GQA layers and 36 fixed recurrent/conv states.
        let state = 36 * (64 * 128 * 128 + 3 * 64 * 128 * 4) * 4;
        for (format, row) in [
            ("hybrid", 1568),
            ("bf16", 4096),
            ("fp8", 2080),
            ("fp4", 1056),
        ] {
            let _format = EnvGuard::set("DS4_SOLAR_KV_FORMAT", format);
            for ctx in [64, 262144] {
                assert_eq!(
                    bank_kv_bytes(SHAPE_SOLAR_OPEN2_250B, ctx, 2048),
                    12 * ctx * row + state,
                    "{format} ctx={ctx}"
                );
            }
        }
    }

    #[test]
    fn motif_bank_uses_latent_windows() {
        // Native includes 14 full layers, 39 SWA layers and one MTP window.
        for ctx in [64u64, 262144] {
            for native in [256, 4096] {
                let rows = 14 * ctx + 40 * ctx.min(128 + 1 + u64::from(native));
                assert_eq!(
                    bank_kv_bytes(SHAPE_MOTIF3, ctx, native),
                    rows * (512 + 64) * 2
                );
            }
        }
    }

    #[test]
    fn motif_checkpoint_stays_fixed() {
        let _env = lock_test_env();
        for ctx in [4096, 262144] {
            let mut req = ServingRequest::default();
            req.ctx = ctx;
            for reuse in [
                PrefixReuse::Auto,
                PrefixReuse::Partial,
                PrefixReuse::Off,
                PrefixReuse::Exact,
            ] {
                req.prefix_reuse = reuse;
                let facts = fill_family(
                    ModelFamily::Motif3,
                    Variant::Motif3,
                    SHAPE_MOTIF3,
                    &req,
                    qwen_host(None),
                );
                let expected = if matches!(reuse, PrefixReuse::Auto | PrefixReuse::Partial) {
                    32 * 39 * 128 * (512 + 64) * 2
                } else {
                    0
                };
                assert_eq!(
                    facts.checkpoint_pool_bytes,
                    Some(expected),
                    "ctx={ctx} reuse={reuse:?}"
                );
            }
        }
    }

    #[test]
    fn qwen_static_skips_auto_mtp() {
        let _env = lock_test_env();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        for (max_seqs, cont_lane, lane) in [
            (MaxSeqs::Auto, Some(false), LaneMode::Auto),
            (MaxSeqs::Off, None, LaneMode::Auto),
            (MaxSeqs::Auto, None, LaneMode::Serial),
        ] {
            let mut req = ServingRequest::default();
            req.max_seqs = max_seqs;
            req.lane = lane;
            let mut off = EngineFacts {
                cont_lane,
                ..EngineFacts::default()
            };
            req.mtp_mode = MtpMode::Off;
            fill_quote_facts(
                &mut off,
                &req,
                caps,
                Some(SHAPE_QWEN38_FLASH_NEXT),
                qwen_host(None),
            );
            let mut auto = EngineFacts {
                cont_lane,
                ..EngineFacts::default()
            };
            req.mtp_mode = MtpMode::Auto;
            fill_quote_facts(
                &mut auto,
                &req,
                caps,
                Some(SHAPE_QWEN38_FLASH_NEXT),
                qwen_host(None),
            );
            assert_eq!(
                resolve_plan(&req, Some(caps), &auto).effective.mtp_mode,
                MtpMode::Off
            );
            assert_eq!(auto.per_bank_bytes, off.per_bank_bytes);
            auto.host_available_bytes = Some(facts_cost(&off, &req, 1));
            let plan = resolve_plan(&req, Some(caps), &auto);
            assert!(!plan.has_errors(), "{:?}", plan.issues);
        }
    }

    #[test]
    fn solar_checkpoint_uses_kda_state_slots() {
        let _env = lock_test_env();
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 0;
        req.prefix_reuse = PrefixReuse::Partial;
        let facts = fill_family(
            ModelFamily::SolarOpen2,
            Variant::SolarOpen2_250B,
            SHAPE_SOLAR_OPEN2_250B,
            &req,
            QuoteHost {
                weights_bytes: 10 * GIB,
                mtp_bytes: 0,
                available_bytes: 100 * GIB,
                native_chunk: None,
                vision: false,
            },
        );
        let kv = bank_kv_bytes(SHAPE_SOLAR_OPEN2_250B, req.ctx.max(1) as u64, 0);
        let pool = facts.checkpoint_pool_bytes.unwrap();
        assert_eq!(pool, solar_checkpoint_pool_bytes(SHAPE_SOLAR_OPEN2_250B));
        assert_ne!(pool, kv);
        assert!(pool > 4 * GIB, "32 KDA state copies are several GiB");
    }

    #[test]
    fn discrete_cuda_uses_device_free_not_ram() {
        let ram = 64 * GIB;
        let avail = 50 * GIB;
        let vram = 24 * GIB;
        let free = 20 * GIB;
        assert_eq!(quote_ceiling(avail, ram, Some((vram, free))), free);
        assert_eq!(quote_ceiling(avail, ram, None), avail);
        assert_eq!(
            quote_ceiling(avail, ram, Some((120 * GIB, 110 * GIB))),
            avail
        );
        assert_eq!(parse_nvidia_csv("[N/A], [N/A]\n"), None);
        assert_eq!(
            parse_nvidia_csv("24576, 20480\n"),
            Some((24576 * MIB, 20480 * MIB))
        );
    }

    #[test]
    fn cpu_quote_uses_ram_not_discrete_fb() {
        let _env = lock_test_env();
        let ram = meminfo_available();
        assert!(ram > 512 * MIB, "need host RAM above the fake FB");

        let dir = std::env::temp_dir().join(format!(
            "ds4-quote-smi-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let smi = dir.join("nvidia-smi");
        std::fs::write(&smi, "#!/bin/sh\necho '1024, 512'\n").unwrap();
        let mut perm = std::fs::metadata(&smi).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            perm.set_mode(0o755);
        }
        std::fs::set_permissions(&smi, perm).unwrap();

        let path = format!(
            "{}:{}",
            dir.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let _path = EnvGuard::set("PATH", &path);
        assert_eq!(host_available_bytes(Backend::Cuda), 512 * MIB);
        let cpu = host_available_bytes(Backend::Cpu);
        let metal = host_available_bytes(Backend::Metal);
        let avail = meminfo_available();
        assert_eq!(cpu, metal);
        assert!(
            cpu.abs_diff(avail) < MIB,
            "CPU {cpu} vs MemAvailable {avail}"
        );
        assert!(cpu > 512 * MIB);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn fill_family(
        family: ModelFamily,
        variant: Variant,
        shape: Shape,
        req: &ServingRequest,
        host: QuoteHost,
    ) -> EngineFacts {
        let mut facts = EngineFacts::default();
        let caps = serving_caps(family, variant);
        fill_quote_facts(&mut facts, req, caps, Some(shape), host);
        facts
    }

    #[test]
    fn deepseek_preserves_chunk_env() {
        let _env = lock_test_env();
        let shape = crate::shape::SHAPE_FLASH;
        let req = ServingRequest::default();
        let caps = serving_caps(shape.family, shape.variant);
        for (value, expected) in [("256", 256), ("512", 512), ("0", 8192)] {
            let _chunk = EnvGuard::set("DS4_METAL_PREFILL_CHUNK", value);
            let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
            assert_eq!(facts.native_chunk, Some(expected));
            let plan = resolve_plan(&req, Some(caps), &facts);
            assert_eq!(plan.batch_max_total_tokens(8192, 2), expected as i32);
            assert!(plan
                .env_overrides()
                .contains(&("DS4_METAL_PREFILL_CHUNK".into(), expected.to_string())));
        }
    }

    #[test]
    fn deepseek_banks_match_native() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset("DS4_METAL_PREFILL_CHUNK");
        let _raw = EnvGuard::unset("DS4_METAL_GRAPH_RAW_CAP");
        for (shape, ctx, expected, mtp_bank) in [
            (crate::shape::SHAPE_FLASH, 8192, 593_119_128, 9_937_932),
            (crate::shape::SHAPE_PRO, 8192, 850_305_176, 10_318_860),
            (crate::shape::SHAPE_FLASH, 262144, 5_199_141_784, 9_937_932),
            (crate::shape::SHAPE_PRO, 262144, 7_443_732_376, 10_318_860),
        ] {
            let mut req = ServingRequest {
                ctx,
                mtp_mode: MtpMode::Off,
                ..ServingRequest::default()
            };
            let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
            assert_eq!(facts.per_bank_bytes, Some(expected));
            assert_eq!(facts.checkpoint_pool_bytes, Some(0));
            req.mtp_path = Some("support.gguf".into());
            let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
            assert_eq!(facts.per_bank_bytes, Some(expected + mtp_bank));
            req.max_seqs = MaxSeqs::Off;
            let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
            assert_eq!(facts.per_bank_bytes, Some(0));
        }
    }

    #[test]
    fn deepseek_dspark_runtime_quoted() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset("DS4_METAL_PREFILL_CHUNK");
        let _raw = EnvGuard::unset("DS4_METAL_GRAPH_RAW_CAP");
        let _off = EnvGuard::set("DS4_CONT_DSPARK", "0");
        for (shape, shared) in [
            (crate::shape::SHAPE_FLASH, 306_148_352),
            (crate::shape::SHAPE_PRO, 507_474_944),
        ] {
            let mut req = ServingRequest::default();
            for width in [MaxSeqs::Auto, MaxSeqs::Off] {
                req.max_seqs = width;
                let caps = serving_caps(shape.family, shape.variant);
                let base = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
                let mut loaded = EngineFacts {
                    dspark_ok: Some(true),
                    ..EngineFacts::default()
                };
                fill_quote_facts(&mut loaded, &req, caps, Some(shape), qwen_host(None));
                assert_eq!(
                    loaded.scratch_bytes.unwrap() - base.scratch_bytes.unwrap(),
                    shared
                );
                let bank = if width == MaxSeqs::Off { 0 } else { 26_738_688 };
                assert_eq!(
                    loaded.per_bank_bytes.unwrap() - base.per_bank_bytes.unwrap(),
                    bank
                );
            }
        }
    }

    #[test]
    fn deepseek_quote_native_graph() {
        let _env = lock_test_env();
        let _raw = EnvGuard::unset("DS4_METAL_GRAPH_RAW_CAP");
        for (shape, ctx, native, expected) in [
            (crate::shape::SHAPE_FLASH, 8192, 4096, 4_958_202_776),
            (crate::shape::SHAPE_PRO, 8192, 4096, 8_238_630_040),
            (crate::shape::SHAPE_FLASH, 262144, 4096, 11_644_600_216),
            (crate::shape::SHAPE_FLASH, 8192, 256, 638_909_336),
        ] {
            let req = ServingRequest {
                ctx,
                native_chunk: Some(native),
                mtp_mode: MtpMode::Off,
                ..ServingRequest::default()
            };
            let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
            assert_eq!(facts.scratch_bytes, Some(expected));
        }
    }

    #[test]
    fn deepseek_loaded_mtp_allocated() {
        let _env = lock_test_env();
        let _raw = EnvGuard::unset("DS4_METAL_GRAPH_RAW_CAP");
        for (shape, expected) in [
            (crate::shape::SHAPE_FLASH, 62_717_952),
            (crate::shape::SHAPE_PRO, 82_231_296),
        ] {
            let req = ServingRequest {
                ctx: 8192,
                native_chunk: Some(4096),
                mtp_mode: MtpMode::Off,
                mtp_path: Some("support.gguf".into()),
                ..ServingRequest::default()
            };
            let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
            assert_eq!(facts.mtp_state_bytes, Some(expected));
        }
    }

    #[test]
    fn glm_quote_matches_native() {
        let _env = lock_test_env();
        let caps = serving_caps(ModelFamily::Glm53, Variant::Glm53Flash);
        let mut req = ServingRequest::default();
        req.ctx = 2048;
        req.mem_floor_gb = 0;
        let mut facts = fill_family(
            ModelFamily::Glm53,
            Variant::Glm53Flash,
            SHAPE_GLM53_FLASH,
            &req,
            qwen_host(None),
        );
        assert_eq!(
            facts.per_bank_bytes.unwrap() + facts.scratch_bytes.unwrap(),
            1_648_433_940
        );
        facts.host_available_bytes = Some(facts_cost(&facts, &req, 1) - 1);
        assert!(resolve_plan(&req, Some(caps), &facts).has_errors());
    }

    #[test]
    fn serial_fit_keeps_headroom() {
        let _env = lock_test_env();
        let _fit = EnvGuard::unset("DS4_SESSION_GRAPH_FIT");
        let _margin = EnvGuard::unset("DS4_SESSION_GRAPH_HEADROOM_MB");
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let mut req = ServingRequest::default();
        req.max_seqs = MaxSeqs::Off;
        req.mem_floor_gb = 0;
        for (key, value, expected) in [
            ("DS4_SESSION_GRAPH_FIT", "1", GIB),
            ("DS4_SESSION_GRAPH_HEADROOM_MB", "2048", 2 * GIB),
            ("DS4_SESSION_GRAPH_FIT", "0", 0),
        ] {
            let _setting = EnvGuard::set(key, value);
            let mut facts = fill_qwen(&req, qwen_host(None));
            assert_eq!(
                resolve_plan(&req, Some(caps), &facts).quote.unwrap().floor,
                expected
            );
            facts.host_available_bytes = Some(facts_cost(&facts, &req, 1) - 1);
            assert!(resolve_plan(&req, Some(caps), &facts).has_errors());
        }
        req.backend = Backend::Cpu;
        let facts = fill_qwen(&req, qwen_host(None));
        assert_eq!(
            resolve_plan(&req, Some(caps), &facts).quote.unwrap().floor,
            0
        );
    }

    #[test]
    fn static_fit_keeps_headroom() {
        let _env = lock_test_env();
        let _headroom = EnvGuard::unset(FIT_HEADROOM_ENV);
        let _derived = EnvGuard::unset(FIT_DERIVED_ENV);
        let _burst = EnvGuard::unset(FIT_BURST_ENV);
        let caps = serving_caps(ModelFamily::Motif3, Variant::Motif3);
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 4;
        for lane in [LaneMode::Auto, LaneMode::Serial] {
            req.lane = lane;
            let mut facts = EngineFacts {
                cont_lane: Some(false),
                ..EngineFacts::default()
            };
            fill_quote_facts(&mut facts, &req, caps, Some(SHAPE_MOTIF3), qwen_host(None));
            let plan = resolve_plan(&req, Some(caps), &facts);
            assert_eq!(plan.quote.unwrap().floor, 6 * GIB);
            facts.host_available_bytes = Some(facts_cost(&facts, &req, 2) - 1);
            assert_eq!(resolve_plan(&req, Some(caps), &facts).effective.max_seqs, 1);
        }
        req.max_seqs = MaxSeqs::Off;
        let facts = fill_family(
            ModelFamily::Motif3,
            Variant::Motif3,
            SHAPE_MOTIF3,
            &req,
            qwen_host(None),
        );
        assert_eq!(
            resolve_plan(&req, Some(caps), &facts).quote.unwrap().floor,
            4 * GIB
        );
    }

    #[test]
    fn dots3_quote_matches_native() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(DOTS3_PREFILL_CHUNK_ENV, "4096");
        let caps = serving_caps(ModelFamily::Dots3Note, Variant::Dots3NotePrev);
        for (ctx, expected) in [(8192, 6_112_078_720u64), (262144, 11_735_591_808)] {
            let mut req = ServingRequest::default();
            req.ctx = ctx;
            req.mem_floor_gb = 0;
            let mut facts = fill_family(
                ModelFamily::Dots3Note,
                Variant::Dots3NotePrev,
                SHAPE_DOTS3_NOTE_PREV,
                &req,
                qwen_host(None),
            );
            let runtime = facts.per_bank_bytes.unwrap() + facts.scratch_bytes.unwrap();
            assert_eq!(runtime, expected, "ctx={ctx}");
            facts.host_available_bytes = Some(facts_cost(&facts, &req, 1) - 1);
            assert!(resolve_plan(&req, Some(caps), &facts).has_errors());
        }
    }

    #[test]
    fn inkling_loaded_off_quote() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(INKLING_PREFILL_CHUNK_ENV, "1024");
        let caps = serving_caps(ModelFamily::Inkling, Variant::InklingSmall);
        let mut req = ServingRequest::default();
        req.ctx = 1024;
        req.mtp_mode = MtpMode::Off;
        for (path, expected) in [
            (None, 766_264_576u64),
            (Some("inkling-mtp.gguf".into()), 1_523_575_296),
        ] {
            req.mtp_path = path;
            let mut facts = fill_family(
                ModelFamily::Inkling,
                Variant::InklingSmall,
                SHAPE_INKLING_SMALL,
                &req,
                qwen_host(None),
            );
            assert_eq!(
                facts.per_bank_bytes.unwrap()
                    + facts.scratch_bytes.unwrap()
                    + facts.mtp_state_bytes.unwrap(),
                expected
            );
            facts.host_available_bytes = Some(facts_cost(&facts, &req, 1) - 1);
            assert!(resolve_plan(&req, Some(caps), &facts).has_errors());
        }
        req.mtp_path = None;
        let mut loaded = EngineFacts {
            mtp_loaded: true,
            ..EngineFacts::default()
        };
        fill_quote_facts(
            &mut loaded,
            &req,
            caps,
            Some(SHAPE_INKLING_SMALL),
            qwen_host(None),
        );
        assert_eq!(loaded.per_bank_bytes, Some(1_523_575_296));
        req.ctx = 16;
        req.native_chunk = Some(1);
        fill_quote_facts(
            &mut loaded,
            &req,
            caps,
            Some(SHAPE_INKLING_SMALL),
            qwen_host(None),
        );
        assert_eq!(loaded.native_chunk, Some(9));
        assert_eq!(loaded.per_bank_bytes, Some(133_007_264));
    }

    #[test]
    fn step_media_native_buffers() {
        let _env = lock_test_env();
        let mut req = ServingRequest::default();
        for (ctx, expected) in [
            (1024, 288_972_672),
            (8192, 406_413_184),
            (16384, 406_413_184),
        ] {
            req.ctx = ctx;
            let no_media = fill_family(
                ModelFamily::Step37,
                Variant::Step37Flash,
                SHAPE_STEP37_FLASH,
                &req,
                qwen_host(None),
            );
            assert_eq!(no_media.media_reserve_bytes, Some(0));
            let mut host = qwen_host(None);
            host.vision = true;
            let facts = fill_family(
                ModelFamily::Step37,
                Variant::Step37Flash,
                SHAPE_STEP37_FLASH,
                &req,
                host,
            );
            assert_eq!(facts.media_reserve_bytes, Some(expected));
        }
    }

    #[test]
    fn step_quote_matches_native() {
        let _env = lock_test_env();
        let caps = serving_caps(ModelFamily::Step37, Variant::Step37Flash);
        for (ctx, cap, expected) in [
            (8192, 4096, 3_464_772_992u64),
            (262144, 2048, 14_451_080_576),
        ] {
            let _chunk = EnvGuard::set(STEP_PREFILL_CHUNK_ENV, &cap.to_string());
            let mut req = ServingRequest::default();
            req.ctx = ctx;
            req.max_seqs = MaxSeqs::Fixed(2);
            let mut facts = fill_family(
                ModelFamily::Step37,
                Variant::Step37Flash,
                SHAPE_STEP37_FLASH,
                &req,
                qwen_host(None),
            );
            assert_eq!(facts.per_bank_bytes, Some(expected), "ctx={ctx} cap={cap}");
            facts.host_available_bytes = Some(facts_cost(&facts, &req, 2) - 1);
            assert!(resolve_plan(&req, Some(caps), &facts).has_errors());
        }
    }

    #[test]
    fn step_loaded_off_keeps_memory() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(STEP_PREFILL_CHUNK_ENV, "4096");
        let caps = serving_caps(ModelFamily::Step37, Variant::Step37Flash);
        let mut req = ServingRequest::default();
        req.max_seqs = MaxSeqs::Fixed(2);
        req.ctx = 8192;
        req.mtp_path = Some("step-mtp.gguf".into());
        let on = fill_family(
            ModelFamily::Step37,
            Variant::Step37Flash,
            SHAPE_STEP37_FLASH,
            &req,
            qwen_host(None),
        );
        assert_eq!(on.per_bank_bytes, Some(6_161_571_072));
        assert_eq!(
            on.checkpoint_pool_bytes,
            Some(32 * (33 * 512 * 4096 + 3 * (512 * 4096 + 4096 * 4)))
        );
        for mode in [MtpMode::Off, MtpMode::Auto] {
            req.mtp_mode = mode;
            req.mtp_draft = Some(0);
            let off = fill_family(
                ModelFamily::Step37,
                Variant::Step37Flash,
                SHAPE_STEP37_FLASH,
                &req,
                qwen_host(None),
            );
            assert_eq!(off.per_bank_bytes, on.per_bank_bytes);
            assert_eq!(off.checkpoint_pool_bytes, on.checkpoint_pool_bytes);
            assert_eq!(
                resolve_plan(&req, Some(caps), &off).effective.mtp_mode,
                MtpMode::Off
            );
        }
        req.mtp_path = None;
        let mut loaded = EngineFacts {
            mtp_loaded: true,
            ..EngineFacts::default()
        };
        fill_quote_facts(
            &mut loaded,
            &req,
            caps,
            Some(SHAPE_STEP37_FLASH),
            qwen_host(None),
        );
        assert_eq!(loaded.per_bank_bytes, on.per_bank_bytes);
        assert_eq!(loaded.checkpoint_pool_bytes, on.checkpoint_pool_bytes);
        req.ctx = 8;
        req.native_chunk = Some(1);
        fill_quote_facts(
            &mut loaded,
            &req,
            caps,
            Some(SHAPE_STEP37_FLASH),
            qwen_host(None),
        );
        assert_eq!(loaded.native_chunk, Some(4));
        assert_eq!(loaded.per_bank_bytes, Some(8_177_472));
    }

    #[test]
    fn step_auto_mtp_without_path_skips_spec() {
        let _env = lock_test_env();
        let mut auto = ServingRequest::default();
        auto.mem_floor_gb = 0;
        let mut off = auto.clone();
        off.mtp_mode = MtpMode::Off;
        let host = QuoteHost {
            weights_bytes: 10 * GIB,
            mtp_bytes: 0,
            available_bytes: 100 * GIB,
            native_chunk: None,
            vision: false,
        };
        let auto_facts = fill_family(
            ModelFamily::Step37,
            Variant::Step37Flash,
            SHAPE_STEP37_FLASH,
            &auto,
            host,
        );
        let off_facts = fill_family(
            ModelFamily::Step37,
            Variant::Step37Flash,
            SHAPE_STEP37_FLASH,
            &off,
            host,
        );
        assert_eq!(auto_facts.per_bank_bytes, off_facts.per_bank_bytes);
    }

    #[test]
    fn exaone_native_defaults_to_runtime_512() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset("DS4_EXAONE_PREFILL_CHUNK");
        let facts = fill_family(
            ModelFamily::ExaoneMoe,
            Variant::Kexaone236B,
            SHAPE_KEXAONE_236B,
            &ServingRequest::default(),
            QuoteHost {
                weights_bytes: GIB,
                mtp_bytes: 0,
                available_bytes: 100 * GIB,
                native_chunk: None,
                vision: false,
            },
        );
        assert_eq!(facts.native_chunk, Some(512));
    }

    #[test]
    fn exaone_kv_matches_native_layer_caps() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset("DS4_EXAONE_PREFILL_CHUNK");
        let mut req = ServingRequest::default();
        req.ctx = 8192;
        req.native_chunk = Some(512);
        req.mem_floor_gb = 0;
        let facts = fill_family(
            ModelFamily::ExaoneMoe,
            Variant::Kexaone236B,
            SHAPE_KEXAONE_236B,
            &req,
            QuoteHost {
                weights_bytes: GIB,
                mtp_bytes: 0,
                available_bytes: 100 * GIB,
                native_chunk: Some(512),
                vision: false,
            },
        );
        let ctx = 8192u32;
        let native = 512u32;
        let n_swa = SHAPE_KEXAONE_236B.n_swa;
        let period = SHAPE_KEXAONE_236B.n_swa_period;
        let n_exec = SHAPE_KEXAONE_236B
            .n_layer
            .saturating_sub(SHAPE_KEXAONE_236B.n_nextn_predict);
        let mut tokens = 0u64;
        for il in 0..n_exec {
            let cap = if period != 0 && (il % period) == period - 1 {
                ctx
            } else {
                n_swa.saturating_add(native).min(ctx)
            };
            tokens = tokens.saturating_add(u64::from(cap));
        }
        let row = 2
            * u64::from(SHAPE_KEXAONE_236B.n_head_kv)
            * u64::from(SHAPE_KEXAONE_236B.n_head_dim)
            * SIZEOF_U16;
        let want = tokens.saturating_mul(row);
        let full = u64::from(n_exec)
            .saturating_mul(u64::from(ctx))
            .saturating_mul(row);
        assert!(want < full, "sliding rings must shrink past 12 full layers");
        assert_eq!(facts.per_bank_bytes, Some(want));
        assert_eq!(facts.native_chunk, Some(512));
    }

    #[test]
    fn k2_native_defaults_to_runtime_1024() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset("DS4_EXAONE_PREFILL_CHUNK");
        let facts = fill_family(
            ModelFamily::ExaoneMoe,
            Variant::K2Horizon375B,
            SHAPE_K2_HORIZON_375B,
            &ServingRequest::default(),
            QuoteHost {
                weights_bytes: GIB,
                mtp_bytes: 0,
                available_bytes: 100 * GIB,
                native_chunk: None,
                vision: false,
            },
        );
        assert_eq!(facts.native_chunk, Some(1024));
    }

    #[test]
    fn motif_native_is_published_to_c_env() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset("DS4_MOTIF3_PREFILL_CHUNK");
        let mut req = ServingRequest::default();
        req.native_chunk = Some(256);
        let facts = fill_family(
            ModelFamily::Motif3,
            Variant::Motif3,
            SHAPE_MOTIF3,
            &req,
            QuoteHost {
                weights_bytes: GIB,
                mtp_bytes: 0,
                available_bytes: 100 * GIB,
                native_chunk: Some(256),
                vision: false,
            },
        );
        assert_eq!(facts.native_chunk, Some(256));
        let caps = serving_caps(ModelFamily::Motif3, Variant::Motif3);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(
            plan.env_overrides()
                .iter()
                .any(|(k, v)| k == "DS4_MOTIF3_PREFILL_CHUNK" && v == "256"),
            "{:?}",
            plan.env_overrides()
        );
    }

    #[test]
    fn solar_native_is_published_to_c_env() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset("DS4_METAL_PREFILL_CHUNK");
        let mut req = ServingRequest::default();
        req.native_chunk = Some(256);
        let facts = fill_family(
            ModelFamily::SolarOpen2,
            Variant::SolarOpen2_250B,
            SHAPE_SOLAR_OPEN2_250B,
            &req,
            QuoteHost {
                weights_bytes: GIB,
                mtp_bytes: 0,
                available_bytes: 100 * GIB,
                native_chunk: Some(256),
                vision: false,
            },
        );
        assert_eq!(facts.native_chunk, Some(256));
        let caps = serving_caps(ModelFamily::SolarOpen2, Variant::SolarOpen2_250B);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(
            plan.env_overrides()
                .iter()
                .any(|(k, v)| k == "DS4_METAL_PREFILL_CHUNK" && v == "256"),
            "{:?}",
            plan.env_overrides()
        );
    }

    #[test]
    fn qwen_native_is_published_to_c_env() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(QWEN_PREFILL_CHUNK_ENV, "8192");
        let mut req = ServingRequest::default();
        req.native_chunk = Some(256);
        let facts = fill_qwen(&req, qwen_host(Some(256)));
        assert_eq!(facts.native_chunk, Some(256));
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(
            plan.env_overrides()
                .iter()
                .any(|(k, v)| k == "DS4_QWEN_PREFILL_CHUNK" && v == "256"),
            "{:?}",
            plan.env_overrides()
        );
    }

    #[test]
    fn qwen_env_allows_wider_yield() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(QWEN_PREFILL_CHUNK_ENV, "1024");
        let mut req = ServingRequest::default();
        req.sched_chunk = Some(512);
        let facts = fill_qwen(&req, qwen_host(None));
        assert_eq!(facts.native_chunk, Some(1024));
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(
            !plan.issues.iter().any(|i| i.code == "chunk_past_native"),
            "{:?}",
            plan.issues
        );
        assert!(plan.may_listen(), "{:?}", plan.issues);
    }

    fn attach_ipc(
        model: &Path,
        mtp: Option<&Path>,
        vision: Option<&Path>,
        dspark: Option<&Path>,
        resident: bool,
    ) -> EngineFacts {
        let mut facts = EngineFacts::default();
        let req = ServingRequest::default();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        attach_host_quote(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            Some(model),
            mtp,
            vision,
            dspark,
            2,
            vision.is_some(),
            resident,
        );
        facts
    }

    #[test]
    fn ipc_manifest_skips_imported_spans() {
        let _env = lock_test_env();
        let dir = std::env::temp_dir().join(format!("ds4-quote-ipc-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let a = dir.join("model-00001-of-00002.gguf");
        let b = dir.join("model-00002-of-00002.gguf");
        let mtp = dir.join("mtp.gguf");
        let vision = dir.join("vision.gguf");
        let dspark = dir.join("dspark.gguf");
        std::fs::File::create(&a)
            .unwrap()
            .write_all(&[0u8; 100])
            .unwrap();
        std::fs::File::create(&b)
            .unwrap()
            .write_all(&[0u8; 40])
            .unwrap();
        std::fs::File::create(&mtp)
            .unwrap()
            .write_all(&[0u8; 25])
            .unwrap();
        std::fs::File::create(&vision)
            .unwrap()
            .write_all(&[0u8; 17])
            .unwrap();
        std::fs::File::create(&dspark)
            .unwrap()
            .write_all(&[0u8; 11])
            .unwrap();
        assert_eq!(gguf_span_bytes(&a, 2), 140);

        {
            let _man = EnvGuard::unset(WEIGHT_IPC_MANIFEST_ENV);
            let facts = attach_ipc(&a, Some(&mtp), Some(&vision), Some(&dspark), false);
            assert_eq!(facts.shared_weights_bytes, Some(140 + 25 + 17 + 11));
        }
        {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::unset(WEIGHT_IPC_SCOPE_ENV);
            let facts = attach_ipc(&a, Some(&mtp), Some(&vision), Some(&dspark), false);
            assert_eq!(facts.shared_weights_bytes, Some(17 + 11));
        }
        {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "both");
            let facts = attach_ipc(&a, Some(&mtp), None, None, false);
            assert_eq!(facts.shared_weights_bytes, Some(0));
        }
        {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "mtp");
            let facts = attach_ipc(&a, Some(&mtp), None, None, false);
            assert_eq!(facts.shared_weights_bytes, Some(140));
        }
        {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "base");
            let facts = attach_ipc(&a, Some(&mtp), None, None, false);
            assert_eq!(facts.shared_weights_bytes, Some(25));
        }

        let live = host_available_bytes(Backend::Cuda);
        if live > 0 {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "both");
            let facts = attach_ipc(&a, Some(&mtp), None, None, true);
            assert_eq!(facts.shared_weights_bytes, Some(0));
            let got = facts.host_available_bytes.unwrap();
            let expect = live.saturating_add(resident_runtime(&facts));
            assert!(
                got.abs_diff(expect) < 256 * MIB,
                "resident ceiling {got} vs {expect}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
