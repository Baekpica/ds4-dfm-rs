//! Host adapter that fills serving-plan quote facts without opening an engine.
//!
//! Weights are the mapped GGUF span. Per-bank / scratch / checkpoint / PLE /
//! media are shape quotes. Available memory is the live host observation.

use std::path::{Path, PathBuf};

use crate::gguf::GgufFile;
use crate::serving::{
    EngineFacts, MtpKind, MtpMode, PrefixReuse, ReuseKind, ServingCaps, ServingRequest,
    DEFAULT_SCHED_CHUNK,
};
use crate::shape::{ModelFamily, Shape, Variant};
use crate::tensors::model_split_sibling_path;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
const DEFAULT_PLE_CACHE_MB: u64 = 2048;
const QWEN_PREFILL_CHUNK_ENV: &str = "DS4_QWEN_PREFILL_CHUNK";
const STEP_PREFILL_CHUNK_ENV: &str = "DS4_STEP37_PREFILL_CHUNK";
const INKLING_PREFILL_CHUNK_ENV: &str = "DS4_INKLING_PREFILL_CHUNK";
const QWEN_NATIVE_DEFAULT: u32 = 256;
const QWEN_NATIVE_MAX: u32 = 16384;
const STEP_NATIVE_DEFAULT: u32 = 4096;
const STEP_NATIVE_MAX: u32 = 4096;
const INKLING_NATIVE_DEFAULT: u32 = 1024;
const INKLING_NATIVE_MAX: u32 = 8192;
const GLM_NATIVE_DEFAULT: u32 = 2048;
const EXAONE_PREFILL_CHUNK_ENV: &str = "DS4_EXAONE_PREFILL_CHUNK";
const EXAONE_NATIVE_DEFAULT: u32 = 512;
const K2_NATIVE_DEFAULT: u32 = 1024;
const MOTIF_PREFILL_CHUNK_ENV: &str = "DS4_MOTIF3_PREFILL_CHUNK";
const MOTIF_NATIVE_DEFAULT: u32 = 4096;
const MOTIF_NATIVE_MAX: u32 = 8192;
const SOLAR_PREFILL_CHUNK_ENV: &str = "DS4_METAL_PREFILL_CHUNK";
const SOLAR_NATIVE_DEFAULT: u32 = 2048;
const DOTS3_PREFILL_CHUNK_ENV: &str = "DS4_DOTS3_PREFILL_CHUNK";
const DOTS3_NATIVE_DEFAULT: u32 = 4096;
const DOTS3_NATIVE_MAX: u32 = 8192;
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
const QWEN_CHECKPOINT_SLOTS: u64 = 32;
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
    let native = host
        .native_chunk
        .or(req.native_chunk)
        .unwrap_or(runtime)
        .min(runtime)
        .max(1);
    let ctx = u64::from(ctx_tokens);
    let kv = shape.map(|s| bank_kv_bytes(s, ctx)).unwrap_or(0);
    // Sidecar MTP is off until a path/load exists; family capability alone
    // would charge Step spec graphs that C will not allocate.
    let mtp_on = match caps.mtp {
        MtpKind::Embedded => req.mtp_mode != MtpMode::Off,
        MtpKind::Sidecar | MtpKind::DeepSeek => {
            req.mtp_mode != MtpMode::Off && (req.mtp_path.is_some() || facts.mtp_loaded)
        }
        MtpKind::BoundOnly | MtpKind::None => false,
    };
    // Native skips the slab when DS4_SERVER_FORK_PARTIAL=0.
    let partial = match req.prefix_reuse {
        PrefixReuse::Off | PrefixReuse::Exact => false,
        PrefixReuse::Partial | PrefixReuse::Auto => caps.reuse == ReuseKind::Partial,
    };
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
            let graph = family_graph_scratch(s, native);
            let spec = if mtp_on { graph } else { 0 };
            let pool = if partial { kv } else { 0 };
            (kv.saturating_add(graph).saturating_add(spec), 0, 0, pool)
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
    let media = if caps.media_serial || host.vision {
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
    let live = host_available_bytes();
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

pub fn host_available_bytes() -> u64 {
    meminfo_available()
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
        ModelFamily::SolarOpen2 => solar_native_chunk(ctx),
        ModelFamily::Dots3Note => env_u32(
            DOTS3_PREFILL_CHUNK_ENV,
            DOTS3_NATIVE_DEFAULT,
            1,
            DOTS3_NATIVE_MAX,
        ),
        _ => DEFAULT_SCHED_CHUNK,
    };
    cap.min(ctx)
}

fn solar_native_chunk(ctx: u32) -> u32 {
    let fallback = ctx.min(SOLAR_NATIVE_DEFAULT).max(1);
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
    qwen_checkpoint_slot_bytes(shape).saturating_mul(QWEN_CHECKPOINT_SLOTS)
}

fn family_graph_scratch(shape: Shape, native: u32) -> u64 {
    u64::from(native)
        .saturating_mul(u64::from(shape.n_embd))
        .saturating_mul(u64::from(shape.n_layer))
        .saturating_mul(2)
}

fn resident_runtime(facts: &EngineFacts) -> u64 {
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

fn bank_kv_bytes(shape: Shape, ctx: u64) -> u64 {
    let row = if shape.n_kv_lora > 0 {
        u64::from(shape.n_kv_lora + shape.n_key_mla + shape.n_value_mla).max(1) * 2
    } else {
        2 * u64::from(shape.n_head_kv.max(1))
            * u64::from(shape.n_head_dim.max(shape.n_value_dim).max(1))
            * 2
    };
    u64::from(shape.n_layer)
        .saturating_mul(ctx)
        .saturating_mul(row)
}

fn ple_cache_bytes(caps: ServingCaps) -> u64 {
    if caps.family != ModelFamily::Qwen4Exp {
        return 0;
    }
    let mb = std::env::var("DS4_QWEN_PLE_CACHE_MB")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(DEFAULT_PLE_CACHE_MB);
    mb.saturating_mul(MIB)
}

fn file_len(path: Option<&Path>) -> u64 {
    path.and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0)
}

fn meminfo_available() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
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
    use crate::serving::{resolve_plan, serving_caps, MtpMode, PrefixReuse, PREFILL_CHUNK_FENCE};
    use crate::shape::{
        Variant, SHAPE_K2_HORIZON_375B, SHAPE_KEXAONE_236B, SHAPE_MOTIF3, SHAPE_QWEN38_FLASH_NEXT,
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
        assert!(host_available_bytes() > 0);
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
            .saturating_add(req.mem_floor_gb.saturating_mul(GIB))
    }

    #[test]
    fn qwen_two_bank_quote_charges_each_graph() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(QWEN_PREFILL_CHUNK_ENV, "8192");
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 0;
        let facts = fill_qwen(&req, qwen_host(None));
        let kv = bank_kv_bytes(SHAPE_QWEN38_FLASH_NEXT, req.ctx.max(1) as u64);
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
        let kv = bank_kv_bytes(SHAPE_QWEN38_FLASH_NEXT, req.ctx.max(1) as u64);
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
    fn qwen_checkpoint_uses_recurrent_slots() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut req = ServingRequest::default();
        req.mem_floor_gb = 0;
        let facts = fill_qwen(&req, qwen_host(None));
        let kv = bank_kv_bytes(SHAPE_QWEN38_FLASH_NEXT, req.ctx.max(1) as u64);
        let pool = facts.checkpoint_pool_bytes.unwrap();
        assert_ne!(pool, kv, "checkpoint slab is recurrent slots, not one KV");
        assert_eq!(pool % QWEN_CHECKPOINT_SLOTS, 0);
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
        let kv = bank_kv_bytes(SHAPE_STEP37_FLASH, req.ctx.max(1) as u64);
        let per_bank = facts.per_bank_bytes.unwrap();
        assert_eq!(facts.scratch_bytes, Some(0));
        assert!(per_bank > kv, "each Step bank owns a graph");
        assert_eq!(
            facts_cost(&facts, &req, 2) - facts_cost(&facts, &req, 1),
            per_bank
        );
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

        let live = host_available_bytes();
        if live > 0 {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "both");
            let facts = attach_ipc(&a, Some(&mtp), None, None, true);
            assert_eq!(facts.shared_weights_bytes, Some(0));
            assert_eq!(
                facts.host_available_bytes,
                Some(live.saturating_add(resident_runtime(&facts)))
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
