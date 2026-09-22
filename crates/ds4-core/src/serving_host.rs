//! Host adapter that fills serving-plan quote facts without opening an engine.
//!
//! Weights are the mapped GGUF span. Per-bank / scratch / checkpoint / PLE /
//! media are shape quotes. Available memory is the live host observation.

use std::path::{Path, PathBuf};

use crate::gguf::GgufFile;
use crate::ling3vl;
use crate::serving::{
    BankLane, EngineFacts, LaneMode, MaxSeqs, MtpKind, MtpMode, PrefixReuse, ReuseKind,
    ServingCaps, ServingRequest, Support, DEFAULT_MAX_SEQS, DEFAULT_SCHED_CHUNK,
};
use crate::shape::{ModelFamily, Shape, Variant};
use crate::tensors::{model_split_sibling_path, TensorInventory};
use crate::Backend;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
// The tensors a sliced model map retains, mirroring native
// `model_map_span_vec_include_layer` / `_include_output`. Every other group
// the artifact carries — vision, MTP, drafter — stays off a sliced map.
const LAYER_TENSOR_PREFIX: &str = "blk.";
const EMBED_TENSOR: &str = "token_embd.weight";
const OUTPUT_TENSORS: [&str; 8] = [
    "output.weight",
    "output_norm.weight",
    "output_hc_base.weight",
    "output_hc_fn.weight",
    "output_hc_scale.weight",
    "hc_input.norm.weight",
    "hc_input.mix_down.weight",
    "hc_input.mix_up.weight",
];
const DEFAULT_PLE_CACHE_MB: u64 = 2048;
const CPU_MAX_THREADS: u64 = 32;
const CPU_FFN_BATCH_MAX: u64 = 4095;
const PLE_CACHE_MB_ENV: &str = "DS4_QWEN_PLE_CACHE_MB";
const PLE_CACHE_MB_512: u64 = 512;
const PLE_CACHE_MB_1024: u64 = 1024;
const QWEN_PREFILL_CHUNK_ENV: &str = "DS4_QWEN_PREFILL_CHUNK";
const STEP_PREFILL_CHUNK_ENV: &str = "DS4_STEP37_PREFILL_CHUNK";
const INKLING_PREFILL_CHUNK_ENV: &str = "DS4_INKLING_PREFILL_CHUNK";
const LING_PREFILL_CHUNK_ENV: &str = "DS4_LING3VL_PREFILL_CHUNK";
const QWEN_NATIVE_DEFAULT: u32 = 256;
const QWEN_NATIVE_MAX: u32 = 16384;
const QWEN_IMAGE_MAX_PIXELS: u64 = 16_777_216;
const QWEN_IMAGE_MAX_COUNT: u64 = 4;
const QWEN_IMAGE_FACTOR: u64 = 32;
const QWEN_IMAGE_MAX_AXIS: u64 = 65536;
const QWEN_VISION_PATCH: u64 = 3 * 2 * 16 * 16;
const QWEN_VISION_HIDDEN: u64 = 1152;
const QWEN_VISION_FF: u64 = 4304;
const LING_NATIVE_DEFAULT: u32 = 2048;
const LING_NATIVE_MAX: u32 = 4096;
const STEP_NATIVE_DEFAULT: u32 = 4096;
const STEP_NATIVE_MAX: u32 = 4096;
const INKLING_NATIVE_DEFAULT: u32 = 2048;
const INKLING_NATIVE_MAX: u32 = 8192;
const INKLING_REL_DIM: u64 = 16;
const INKLING_GLOBAL_ROWS: u64 = 1024;
const INKLING_DRAFT_GLOBALS: u64 = 2;
const INKLING_MEDIA_ROWS: u64 = 8192;
const INKLING_MEDIA_INPUTS: u64 = 4;
const INKLING_IMAGE_VALUES: u64 = 2 * 40 * 40 * 3;
const INKLING_DECODE_LIMIT: u64 = 128 * MIB;
const STEP_VISION_EDGE: u64 = 728;
const STEP_VISION_PATCH: u64 = 14;
const STEP_VISION_DIM: u64 = 1536;
const STEP_VISION_FFN: u64 = 8960;
const STEP_MEDIA_ROWS: u64 = 8192;
const STEP_RGB_LIMIT: u64 = 128 * MIB;
const STEP_SOURCE_EDGE: u64 = 3024;
const STEP_PIXELS_PER_TOKEN: u64 = 56 * 56 * 3;
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
const MIMO_PREFILL_CHUNK_ENV: &str = "DS4_MIMO2_PREFILL_CHUNK";
const MIMO_NATIVE_DEFAULT: u32 = crate::mimo2::PREFILL_CAP;
const MIMO_NATIVE_MAX: u32 = crate::mimo2::PREFILL_CAP;
const DOTS3_INDEX_ROWS: u64 = 128;
const DOTS3_PARTIAL_ROWS: u64 = 2;
const DOTS3_PARTIAL_SPLITS: u64 = 16;
const DOTS3_TRIAL_ROWS: u64 = 4;
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
// C LING3VL_MEDIA_ROWS, and L3V_VIT_MAX_PATCHES = 4 * ROWS / INPUTS = the same
// number for four inputs.
const LING_MEDIA_ROWS: u32 = 16_384;
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
    // The resolved CLI cap is published to the allocator's family env.
    // Defaults and existing env apply only when no explicit CLI cap is set.
    let ctx_tokens = req.ctx.max(1) as u32;
    let runtime = family_native_chunk(caps, ctx_tokens);
    let mut native = match req.native_chunk {
        Some(n) => n.min(family_native_limit(caps)).min(ctx_tokens),
        None => host.native_chunk.unwrap_or(runtime).min(runtime),
    }
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
            let scratch = solar_graph_bytes(s, ctx, native, req.backend);
            let pool = if partial {
                solar_checkpoint_pool_bytes(s)
            } else {
                0
            };
            let bank_decode = if quote_batch_alloc(req, caps, facts) {
                u64::from(s.n_vocab) * SIZEOF_F32
                    + solar_split_bytes(s, ctx, req.backend)
                    + 2 * SIZEOF_U32
            } else {
                0
            };
            (kv + bank_decode, scratch, 0, pool)
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
        (ModelFamily::Ling3Vl, Some(s)) => {
            let pool = if partial {
                ling_checkpoint_pool_bytes(s)
            } else {
                0
            };
            let bank = ling_latent_bytes(s, ctx)
                .saturating_add(ling_state_bytes(s))
                .saturating_add(ling_graph_bytes(s, native));
            (bank, 0, 0, pool)
        }
        (ModelFamily::Inkling, Some(s)) => {
            let (base, with_mtp) = inkling_runtime_bytes(s, ctx, native);
            (if sidecar_loaded { with_mtp } else { base }, 0, 0, 0)
        }
        (ModelFamily::Dots3Note, Some(s)) => {
            let logits = if quote_batch_alloc(req, caps, facts) {
                u64::from(s.n_vocab) * SIZEOF_F32
            } else {
                0
            };
            (
                dots3_graph_bytes(s, ctx, native) + logits,
                0,
                if req.mtp_mode == MtpMode::On && !quote_bank_lane(req, caps, facts) {
                    dots3_mtp_bytes(s, ctx, native)
                } else {
                    0
                },
                if partial {
                    dots3_checkpoint_bytes(s, ctx)
                } else {
                    0
                },
            )
        }
        (ModelFamily::Glm53, Some(s)) => (glm_graph_bytes(s, ctx), 0, 0, 0),
        (ModelFamily::Mimo2, Some(_)) => {
            let cap = native.min(ctx_tokens).max(1);
            let bytes = crate::mimo2::context_bytes(ctx_tokens, cap).unwrap_or(0);
            (bytes, 0, 0, 0)
        }
        (ModelFamily::DeepSeek4, Some(s)) if req.backend == Backend::Cpu => {
            let (cache, scratch) = deepseek_cpu_bytes(s, ctx);
            (cache, scratch + deepseek_cpu_prefill(s, ctx, scratch), 0, 0)
        }
        (ModelFamily::DeepSeek4, Some(s)) => {
            // The shared graph owns its own caches before bank slabs are fitted.
            let mut scratch = deepseek_graph_bytes(s, ctx, native);
            let mut bank = deepseek_bank_bytes(s, ctx, native);
            let wanted = match req.max_seqs {
                MaxSeqs::Fixed(n) => n,
                _ => caps.qualified_banks.unwrap_or(DEFAULT_MAX_SEQS),
            };
            let banks = u64::from(facts.banks_fitted.unwrap_or(wanted).min(wanted).max(1));
            bank += deepseek_page_extra(s, ctx, banks).div_ceil(banks);
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
        (ModelFamily::ExaoneMoe, Some(s)) => {
            let row = plain_graph_row_elems(s) * SIZEOF_F32;
            let logits = u64::from(s.n_vocab) * SIZEOF_F32;
            let batch_logits = if quote_batch_alloc(req, caps, facts) {
                logits
            } else {
                0
            };
            (
                kv + row + logits + batch_logits,
                u64::from(native) * row,
                0,
                if partial {
                    exaone_checkpoint_bytes(s, ctx)
                } else {
                    0
                },
            )
        }
        _ => (kv, 0, 0, 0),
    };
    let mut media_extra = 0;
    let media = if caps.family == ModelFamily::Qwen4Exp {
        let (reserve, extra) = shape.map(|s| qwen_media_bytes(s, req)).unwrap_or((GIB, 0));
        media_extra = extra;
        reserve
    } else if caps.family == ModelFamily::Inkling {
        if req.backend == Backend::Cuda {
            shape.map(|s| inkling_media_bytes(s, ctx)).unwrap_or(GIB)
        } else {
            0
        }
    } else if caps.family == ModelFamily::Step37 {
        if host.vision || facts.vision_loaded {
            shape.map(|s| step_media_bytes(s, ctx)).unwrap_or(GIB)
        } else {
            0
        }
    } else if caps.family == ModelFamily::Ling3Vl {
        if host.vision || facts.vision_loaded {
            shape
                .map(|s| {
                    let vision = ling_media_bytes(s, ctx);
                    // Persistent banks keep their graphs. The first image
                    // still calls ling3vl_graph_alloc (C ling3vl_session_bytes).
                    if quote_bank_lane(req, caps, facts)
                        && !matches!(req.max_seqs, MaxSeqs::Off | MaxSeqs::Fixed(1))
                    {
                        vision
                            + ling_latent_bytes(s, ctx)
                            + ling_state_bytes(s)
                            + ling_graph_bytes(s, native)
                    } else {
                        vision
                    }
                })
                .unwrap_or(GIB)
        } else {
            0
        }
    } else if caps.family == ModelFamily::Glm53 {
        if host.vision || facts.vision_loaded {
            glm_media_bytes()
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
    facts.media_per_extra_bank_bytes = Some(media_extra);
    facts.fit_headroom_bytes = Some(quote_fit_headroom(req, caps, facts));
    // Zero is an unsupported probe, not a host with no RAM.
    facts.host_available_bytes = (host.available_bytes > 0).then_some(host.available_bytes);
    facts.native_chunk = Some(native);
}

/// Layer interval a distributed slice keeps resident.
///
/// Mirrors the native `weights_model_map_spans` selection: layer 0 pulls the
/// token embedding in, and `output` adds the final norm/head tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightSlice {
    pub start: u32,
    /// Inclusive last layer; `u32::MAX` runs to the last block.
    pub end: u32,
    pub output: bool,
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
    slice: Option<WeightSlice>,
    vision: bool,
    resident: bool,
) {
    // Only the base weights are sliced; MTP and drafter load whole.
    let mut weights_bytes = model_path
        .map(|path| match slice {
            Some(slice) => gguf_slice_span_bytes(path, split_count, slice),
            None => gguf_span_bytes(path, split_count),
        })
        .unwrap_or(0);
    let mut mtp_bytes = artifact_span_bytes(mtp_path);
    let vision_bytes = artifact_span_bytes(vision_path);
    let mut dspark_bytes = artifact_span_bytes(dspark_path);
    let drafter_pending = req.backend == Backend::Cuda
        && !resident
        && facts.drafter_shared.is_none()
        && ipc_drafter_planned(dspark_bytes);
    if req.backend == Backend::Cuda && facts.drafter_shared == Some(true) {
        dspark_bytes = 0;
    }
    let ipc_pending = req.backend == Backend::Cuda
        && !resident
        && std::env::var_os(WEIGHT_IPC_MANIFEST_ENV).is_some_and(|v| !v.is_empty());
    facts.ipc_pending = ipc_pending || drafter_pending;
    // Native aborts on failed base/MTP imports. Only a successful open
    // proves these requested scopes are shared; a filename is not proof.
    let skip = if req.backend == Backend::Cuda && resident {
        ipc_weight_skip()
    } else {
        IpcSkip::None
    };
    match skip {
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
    // Defer import-dependent admission until open confirms ownership.
    // Check-config reports this as unsupported because it cannot import.
    let live = if facts.ipc_pending {
        0
    } else {
        host_available_bytes(req.backend)
    };
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

// Detect an unconfirmed pre-open import. Native drafter imports can fail
// softly, so this suspends the quote rather than claiming shared ownership.
fn ipc_drafter_planned(model_size: u64) -> bool {
    if model_size == 0 || std::env::var_os("DS4_CUDA_WEIGHT_IPC_NO_DRAFTER").is_some() {
        return false;
    }
    let Some(text) = std::env::var_os(WEIGHT_IPC_MANIFEST_ENV)
        .and_then(|path| std::fs::read_to_string(path).ok())
    else {
        return false;
    };
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));
    let vmm = match lines.next() {
        Some(
            "DS4_WEIGHT_SERVER_IPC_V1" | "DS4_WEIGHT_SERVER_IPC_DERIVED_V1" | "DS4_WEIGHTD_IPC_V1",
        ) => false,
        Some("DS4_WEIGHT_SERVER_VMM_V1" | "DS4_WEIGHT_SERVER_VMM_DERIVED_V1") => true,
        _ => return false,
    };
    lines.any(|line| {
        let fields: Vec<_> = line.split_whitespace().collect();
        let record = if vmm { "alloc" } else { "range" };
        let id = if vmm { 2 } else { 1 };
        if fields.len() != id + 5 || fields[0] != record || fields[id] != "drafter" {
            return false;
        }
        let size = fields[id + 1].parse::<u64>().ok();
        let offset = fields[id + 2].parse::<u64>().ok();
        let bytes = fields[id + 3].parse::<u64>().ok();
        size == Some(model_size)
            && offset.zip(bytes).is_some_and(|(off, len)| {
                len > 0 && off.checked_add(len).is_some_and(|end| end <= model_size)
            })
    })
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

/// Bytes one distributed slice keeps resident.
///
/// A sliced boot restricts the model map to its own layer interval
/// (`weights_model_map_spans`), so pricing the whole sharded artifact
/// rejects a model larger than one GPU even when the slice fits. Spans
/// never overlap, so the selected tensor bytes are the mapped span.
/// An unreadable directory, or a selection that matches nothing, falls back
/// to the full artifact rather than under-pricing the boot.
pub fn gguf_slice_span_bytes(path: &Path, split_count: u32, slice: WeightSlice) -> u64 {
    let Ok(inventory) = TensorInventory::open(path) else {
        return gguf_span_bytes(path, split_count);
    };
    let mut span = 0u64;
    for tensor in &inventory.tensors {
        if slice_holds(&tensor.name, slice) {
            span = span.saturating_add(tensor.bytes);
        }
    }
    // An empty selection means the naming assumption missed, not a free slice.
    if span == 0 {
        return gguf_span_bytes(path, split_count);
    }
    span
}

/// `blk.N.*` belongs to layer N, the token embedding rides with layer 0, and
/// the output group is the native allowlist — not "whatever is left", which
/// would charge the vision and MTP groups to whoever owns the head.
fn slice_holds(name: &str, slice: WeightSlice) -> bool {
    let Some(rest) = name.strip_prefix(LAYER_TENSOR_PREFIX) else {
        if name == EMBED_TENSOR {
            return slice.start == 0;
        }
        return slice.output && OUTPUT_TENSORS.contains(&name);
    };
    let Some(layer) = rest.split('.').next().and_then(|n| n.parse::<u32>().ok()) else {
        return false;
    };
    layer >= slice.start && layer <= slice.end
}

pub fn host_available_bytes(backend: Backend) -> u64 {
    let avail = meminfo_available();
    match backend {
        Backend::Cuda => quote_device(avail, crate::serving_cuda::device()),
        Backend::Metal | Backend::Cpu => avail,
    }
}

fn quote_device(avail: u64, device: Option<crate::serving_cuda::Device>) -> u64 {
    let Some(device) = device else {
        return 0;
    };
    if device.integrated {
        return avail;
    }
    quote_ceiling(nvidia_fb_probe(&device.uuid))
}

// Device identity/topology comes from CUDA, never relative RAM/VRAM capacity.
fn quote_ceiling(device: Option<(u64, u64)>) -> u64 {
    // Zero leaves host_available_bytes unset: RAM cannot price an unknown GPU.
    device.map(|(_, free)| free).unwrap_or(0)
}

fn nvidia_fb_probe(uuid: &str) -> Option<(u64, u64)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .arg(format!("--id={uuid}"))
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

fn family_native_limit(caps: ServingCaps) -> u32 {
    match caps.family {
        ModelFamily::Mimo2 => MIMO_NATIVE_MAX,
        ModelFamily::Qwen4Exp => QWEN_NATIVE_MAX,
        ModelFamily::Step37 => STEP_NATIVE_MAX,
        ModelFamily::Ling3Vl => LING_NATIVE_MAX,
        ModelFamily::Inkling => INKLING_NATIVE_MAX,
        ModelFamily::Glm53 => GLM_NATIVE_DEFAULT,
        ModelFamily::ExaoneMoe => FAMILY_NATIVE_MAX,
        ModelFamily::Motif3 => MOTIF_NATIVE_MAX,
        ModelFamily::Dots3Note => DOTS3_NATIVE_MAX,
        ModelFamily::SolarOpen2 | ModelFamily::DeepSeek4 => u32::MAX,
    }
}

fn family_native_chunk(caps: ServingCaps, ctx: u32) -> u32 {
    let ctx = ctx.max(1);
    let cap = match caps.family {
        ModelFamily::Mimo2 => env_u32(
            MIMO_PREFILL_CHUNK_ENV,
            MIMO_NATIVE_DEFAULT,
            1,
            MIMO_NATIVE_MAX,
        ),
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
        ModelFamily::Ling3Vl => env_u32(
            LING_PREFILL_CHUNK_ENV,
            LING_NATIVE_DEFAULT,
            1,
            LING_NATIVE_MAX,
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

// C `ling3vl_memory` raw_bytes: only the 7 MLA blocks hold a per-token latent
// cache; the 35 recurrent blocks own a context-free tile and three conv rings.
fn ling_state_bytes(s: Shape) -> u64 {
    let kda_dim = u64::from(s.n_head).saturating_mul(u64::from(s.n_kda_head_dim.max(1)));
    let per = kda_dim
        .saturating_mul(u64::from(s.n_kda_head_dim.max(1)))
        .saturating_add(3 * kda_dim * u64::from(s.n_ssm_conv.max(1)))
        .saturating_mul(SIZEOF_F32);
    per.saturating_mul(ling_kda_layers(s))
}

fn ling_kda_layers(s: Shape) -> u64 {
    (0..s.n_layer)
        .filter(|il| ling3vl::layer_is_kda(*il))
        .count() as u64
}

// C `ling3vl_graph_alloc` control_pool: conv weights, decay, dt_bias and
// o_norm for each KDA block. Uploaded once; not part of the checkpoint slab.
fn ling_control_bytes(s: Shape) -> u64 {
    let heads = u64::from(s.n_head);
    let kda_head = u64::from(s.n_kda_head_dim.max(1));
    let kda_dim = heads.saturating_mul(kda_head);
    let conv = kda_dim
        .saturating_mul(u64::from(s.n_ssm_conv.max(1)))
        .saturating_mul(SIZEOF_F32);
    (conv.saturating_mul(3)
        + heads.saturating_mul(SIZEOF_F32)
        + kda_dim.saturating_mul(SIZEOF_F32)
        + kda_head.saturating_mul(SIZEOF_F32))
    .saturating_mul(ling_kda_layers(s))
}

fn ling_latent_bytes(s: Shape, ctx: u64) -> u64 {
    let row = u64::from(s.n_kv_lora).saturating_add(u64::from(s.n_rot));
    let mla = u64::from(s.n_layer).saturating_sub(ling_kda_layers(s));
    mla.saturating_mul(ctx)
        .saturating_mul(row)
        .saturating_mul(SIZEOF_U16)
}

// C `ling3vl_memory` scratch_bytes: one complete graph per bank, so two banks
// price two workspaces rather than one shared scratch.
fn ling_graph_bytes(s: Shape, native: u32) -> u64 {
    let pc = u64::from(native);
    let hidden = u64::from(s.n_embd);
    let heads = u64::from(s.n_head);
    let kda_head = u64::from(s.n_kda_head_dim.max(1));
    let kda_dim = heads * kda_head;
    let q_dim = heads * u64::from(s.n_key_mla);
    let latent = heads * u64::from(s.n_kv_lora);
    let kv_row = u64::from(s.n_kv_lora) + u64::from(s.n_rot);
    let used = u64::from(s.n_expert_used);
    let ff = u64::from(s.n_ff_exp);
    let common = 5 * hidden
        + 2 * q_dim
        + 2 * kda_dim
        + 3 * u64::from(s.n_ff_dense)
        + 3 * ff
        + u64::from(s.n_expert)
        + 2 * used
        + 3 * used * ff
        + used * hidden;
    let family = kv_row + u64::from(s.n_kv_lora) + 2 * latent + 2 * kda_dim + 2 * heads + 4;
    let pairs = u64::from(s.n_rot) / 2;
    (pc * (common + family) + pairs + u64::from(s.n_vocab)) * SIZEOF_F32
        + kda_prefill_scratch_bytes(pc, heads, kda_head)
        + ling_control_bytes(s)
}

// C `ling3vl_ckpt_init`: 32 slots of the recurrent state, reserved and mapped
// on demand. The latent caches need no slot; they are append-only.
fn ling_checkpoint_pool_bytes(s: Shape) -> u64 {
    ling_state_bytes(s).saturating_mul(CHECKPOINT_SLOTS)
}

// C `ling3vl_session_bytes` vision terms: the ViT workspace for a full
// 16,384-patch budget plus the F32 projected-row plane the session fills.
// Images run on the serial lane, so this is reserved once, not per bank.
// When banks stay live, fill_quote_facts adds one language graph on top.
fn ling_media_bytes(s: Shape, ctx: u64) -> u64 {
    let patches = u64::from(LING_MEDIA_ROWS);
    let tower = patches
        * (u64::from(3 * ling3vl::VISION_PATCH * ling3vl::VISION_PATCH)
            + 6 * ling3vl::VISION_EMBED
            + ling3vl::VISION_FF
            + 10)
        + patches / 4 * u64::from(s.n_embd);
    let rows = ctx.min(u64::from(LING_MEDIA_ROWS));
    (tower + rows * u64::from(s.n_embd)) * SIZEOF_F32
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
// EXAONE LLLG snapshots contain the 36 local GQA windows; global layers
// stay in the source bank. The appended MTP block is not executed here.
fn exaone_checkpoint_bytes(shape: Shape, ctx: u64) -> u64 {
    if shape.variant != Variant::Kexaone236B || shape.n_swa_period == 0 {
        return 0;
    }
    let n_exec = shape.n_layer.saturating_sub(shape.n_nextn_predict);
    let local = (0..n_exec)
        .filter(|il| il % shape.n_swa_period != shape.n_swa_period - 1)
        .count() as u64;
    local
        * ctx.min(u64::from(shape.n_swa))
        * 2
        * u64::from(shape.n_head_kv)
        * u64::from(shape.n_head_dim)
        * SIZEOF_U16
        * CHECKPOINT_SLOTS
}

// dots3 snapshots only local MLA latent/RoPE windows. Full MLA and F32
// indexer keys remain in the source bank; the MTP block owns no state.
fn dots3_checkpoint_bytes(shape: Shape, ctx: u64) -> u64 {
    let local = (0..shape.n_layer.saturating_sub(shape.n_nextn_predict))
        .filter(|il| *il != 0 && (shape.n_swa_period == 0 || il % shape.n_swa_period != 1))
        .count() as u64;
    local
        * ctx.min(u64::from(shape.n_swa))
        * u64::from(shape.n_swa_kv_lora + shape.n_rot)
        * SIZEOF_U16
        * CHECKPOINT_SLOTS
}

fn quote_partial(req: &ServingRequest, caps: ServingCaps, facts: &EngineFacts) -> bool {
    if caps.reuse != ReuseKind::Partial
        || (req.prefix_reuse == PrefixReuse::Auto && caps.reuse_support != Support::Qualified)
    {
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

// C kv_cache_init + cpu_decode_scratch_init + the session logits row.
fn deepseek_cpu_bytes(s: Shape, ctx: u64) -> (u64, u64) {
    let dim = u64::from(s.n_head_dim);
    let index = u64::from(s.n_indexer_head_dim);
    let raw = u64::from(s.n_swa).min(ctx).max(1);
    let mut cache = u64::from(s.n_layer) * raw * dim * SIZEOF_F32;
    for il in 0..s.n_layer {
        let ratio = deepseek_comp_ratio(s, il);
        if ratio == 0 {
            continue;
        }
        let cap = ctx / ratio + 2;
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = dim + if ratio == 4 { index } else { 0 };
        cache += (cap * width + 2 * coff * coff * width * ratio) * SIZEOF_F32;
    }
    let hidden = u64::from(s.n_embd);
    let hc = u64::from(s.n_hc);
    let q = u64::from(s.n_head) * dim;
    let ff = u64::from(s.n_ff_exp);
    let used = u64::from(s.n_expert_used);
    let comp = ctx / 4 + 2;
    let floats = 11 * hidden
        + 6 * hc * hidden
        + 2 * q
        + 2 * u64::from(s.n_lora_q)
        + 8 * dim
        + index
        + u64::from(s.n_out_group) * u64::from(s.n_lora_o)
        + raw
        + 2 * comp
        + u64::from(s.n_indexer_head) * (index + 1)
        + (3 + used) * ff
        + 2 * hc
        + u64::from(s.n_vocab);
    // block_q8_K is 292 bytes for 256 values; q8_xq uses 32 bytes + F32 scale.
    let quant = (hidden / 256 + used * (ff / 256)) * 292 + q.div_ceil(32) * 36;
    (cache, floats * SIZEOF_F32 + comp + quant)
}

// CPU prefill holds full-prompt HC rows, independent of the GPU native cap.
// Bound the attention and FFN peaks, including optional parallel-prefix masks
// and all 32 native workers. Per-worker decode scratch conservatively covers
// the smaller token/compressor/indexer temporaries across CPU debug paths.
fn deepseek_cpu_prefill(s: Shape, ctx: u64, decode: u64) -> u64 {
    let hidden = u64::from(s.n_embd);
    let hc = u64::from(s.n_hc);
    let dim = u64::from(s.n_head_dim);
    let q = u64::from(s.n_head) * dim;
    let rank = u64::from(s.n_lora_q);
    let ff = u64::from(s.n_ff_exp);
    let used = u64::from(s.n_expert_used);
    // layer_grouped_out_batch uses eight groups of rank 1024 on CPU.
    let low = 8 * 1024;
    let attn_row = 3 * hidden + hc * hidden + 2 * rank + 2 * q + 2 * dim + hc + hc * hc + low;
    let attn = ctx
        * (attn_row * SIZEOF_F32
            + q.max(hidden).max(rank).max(low).div_ceil(32) * 36
            + (ctx / 4 + 2).div_ceil(8)
            + 5);
    let ffn_rows = ctx * (4 * hidden + hc + hc * hc) * SIZEOF_F32;
    let shared = ctx * (3 * ff * SIZEOF_F32 + hidden.max(ff).div_ceil(32) * 36);
    // Routed batch holds selected/weight/pairs/ids, F32 mid and Q8_K mirrors.
    let routed = ctx.min(CPU_FFN_BATCH_MAX)
        * (used * (ff * SIZEOF_F32 + 20) + (hidden / 256 + used * (ff / 256)) * 292);
    let ffn = ffn_rows + shared.max(routed).max(ctx * hidden * SIZEOF_F32);
    3 * ctx * hc * hidden * SIZEOF_F32 + attn.max(ffn) + CPU_MAX_THREADS * decode
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

// Match native's max(virtual capacity, banded active-page envelope). CUDA
// uses separate slab reservations; only banks inside one slab share pages.
// Spark's 2 MiB pages conservatively cover devices with smaller VMM pages.
fn deepseek_page_extra(s: Shape, ctx: u64, banks: u64) -> u64 {
    if std::env::var("DS4_BATCH_VMM_COMP").as_deref() == Ok("0")
        || env_nonnegative_mb("DS4_BATCH_SLAB_POISON").unwrap_or(0) >= 1
    {
        return 0;
    }
    let packed_on = |key| {
        let value = std::env::var(key).unwrap_or_default();
        !(value.starts_with('0')
            || matches!(
                value.as_str(),
                "off" | "OFF" | "no" | "NO" | "false" | "FALSE"
            ))
    };
    let dim = u64::from(s.n_head_dim);
    let index = u64::from(s.n_indexer_head_dim);
    let packed = dim - u64::from(s.n_rot) + u64::from(s.n_rot) * SIZEOF_F32;
    let comp_row = if packed_on("DS4_CUDA_FP8_KV") {
        packed
    } else {
        dim * SIZEOF_F32
    };
    let index_row = if packed_on("DS4_CUDA_FP4_INDEX") {
        index / 2
    } else {
        index * SIZEOF_F32
    };
    let page = 2 * MIB;
    let mut virtual_bytes = 0;
    let mut page_bytes = 0;
    for il in 0..s.n_layer {
        let ratio = deepseek_comp_ratio(s, il);
        if ratio == 0 {
            continue;
        }
        let rows = banks * (ctx / ratio + 2);
        virtual_bytes += rows * (dim * SIZEOF_F32 + packed);
        page_bytes += (rows * comp_row).div_ceil(page) * page;
        if ratio == 4 {
            virtual_bytes += rows * (index * SIZEOF_F32 + index / 2);
            page_bytes += (rows * index_row).div_ceil(page) * page;
        }
    }
    let band = env_nonnegative_mb("DS4_CONT_ADMIT_BAND_X1024")
        .filter(|n| *n > 0)
        .unwrap_or(1045)
        .clamp(1024, 2048);
    (page_bytes * band)
        .div_ceil(1024)
        .saturating_sub(virtual_bytes)
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

// Shared geometry from the native Solar/EXAONE plain GQA MoE workspace.
fn plain_graph_row_elems(s: Shape) -> u64 {
    let hidden = u64::from(s.n_embd);
    let used = u64::from(s.n_expert_used);
    5 * hidden
        + 2 * u64::from(s.n_head) * u64::from(s.n_head_dim)
        + 2 * u64::from(s.n_head_kv) * u64::from(s.n_head_dim)
        + 3 * u64::from(s.n_ff_dense)
        + 3 * u64::from(s.n_ff_shexp)
        + u64::from(s.n_expert)
        + 2 * used
        + 3 * used * u64::from(s.n_ff_exp)
        + used * hidden
}

// C solar_graph_context_memory_estimate, excluding bank-owned KV/KDA state.
fn solar_graph_bytes(s: Shape, ctx: u64, native: u32, backend: Backend) -> u64 {
    let pc = u64::from(native);
    let heads = u64::from(s.n_head);
    let dim = u64::from(s.n_kda_head_dim);
    let kda = heads * dim;
    let conv = kda * u64::from(s.n_ssm_conv);
    let n_kda = (0..s.n_layer).filter(|il| il % 4 != 0).count() as u64;
    let controls = n_kda * (3 * conv + heads + kda + dim);
    let row = plain_graph_row_elems(s) + 3 * kda + dim + heads;
    ((pc + 1) * row + u64::from(s.n_vocab) + controls) * SIZEOF_F32
        + solar_split_bytes(s, ctx, backend)
        + pc * SIZEOF_I32
        + kda_prefill_scratch_bytes(pc, heads, dim)
}

// C `ds4_gpu_solar_kda_prefill_scratch_bytes`: the chunked delta-rule path
// owns six 256-aligned planes and one per-chunk score tile. Zero means the
// shape or a short append falls back to the generic sequence path.
fn kda_prefill_scratch_bytes(tokens: u64, heads: u64, head_dim: u64) -> u64 {
    if tokens < 64 || heads == 0 || head_dim != 128 {
        return 0;
    }
    let plane = (tokens * heads * head_dim * SIZEOF_F32).div_ceil(256) * 256;
    let mq = (tokens.div_ceil(64) * heads * 64 * 64 * SIZEOF_F32).div_ceil(256) * 256;
    6 * plane + mq
}

fn solar_split_bytes(s: Shape, ctx: u64, backend: Backend) -> u64 {
    let baseline = backend != Backend::Cuda
        || std::env::var("DS4_CUDA_SOLAR_GQA_GROUPED").ok().as_deref() == Some("0");
    let chunk = if baseline {
        2048
    } else {
        std::env::var("DS4_CUDA_SOLAR_GQA_CHUNK")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| matches!(v, 64 | 128 | 256 | 512 | 1024 | 2048))
            .unwrap_or(64)
    };
    u64::from(s.n_head) * ctx.div_ceil(chunk) * (u64::from(s.n_head_dim) + 2) * SIZEOF_F32
}

// prepare_media retains all normalized images through synchronous native
// encoding/prefill. Bound four inputs and the image-or-audio span by context.
fn inkling_media_bytes(s: Shape, ctx: u64) -> u64 {
    let rows = ctx.min(INKLING_MEDIA_INPUTS * INKLING_MEDIA_ROWS);
    let pixels = rows * INKLING_IMAGE_VALUES * SIZEOF_F32;
    let projected = rows * u64::from(s.n_embd) * SIZEOF_F32;
    let pointers = ctx * std::mem::size_of::<usize>() as u64;
    // Two native image buffers, each 16 patches of 2*8*8*128 floats.
    let encoder = rows.min(16) * 2 * (2 * 8 * 8 * 128) * SIZEOF_F32;
    // Decoder working/output buffers or EXIF/RGB copies can overlap, each
    // bounded by 128 MiB. They finish before projected features are created.
    // Audio's 80 codes/frame, waveform/FFT and encoder fit under these bounds.
    pixels + (2 * INKLING_DECODE_LIMIT).max(projected + pointers + encoder)
}

// Embedded vision is always available on Qwen CUDA, with up to four images.
// Match qwen4exp_graph_prepare_images, bounding image rows by prompt capacity.
fn qwen_media_bytes(s: Shape, req: &ServingRequest) -> (u64, u64) {
    let ctx = req.ctx.max(1) as u64;
    if req.backend != Backend::Cuda || ctx < 64 {
        return (0, 0);
    }
    let tokens_per_image = QWEN_IMAGE_MAX_PIXELS / QWEN_IMAGE_FACTOR.pow(2);
    let features = ctx.min(QWEN_IMAGE_MAX_COUNT * tokens_per_image);
    let patches = 4 * features;
    let row = 2 * QWEN_VISION_PATCH + 5 * QWEN_VISION_HIDDEN + QWEN_VISION_FF + 24;
    // Pixel/aspect limits bound every axis below 65536. Nearest-32 rounding
    // can add at most 16 columns to the horizontal resize intermediate.
    let horizontal = QWEN_IMAGE_MAX_PIXELS + (QWEN_IMAGE_FACTOR / 2) * QWEN_IMAGE_MAX_AXIS;
    let resize = 9 * QWEN_IMAGE_MAX_PIXELS + 3 * horizontal + 32 * 4 * QWEN_IMAGE_MAX_AXIS;
    let projected = features * u64::from(s.n_embd) * SIZEOF_F32;
    // Encoding is synchronous. The solver scales other banks' retained
    // features/M-RoPE with each candidate width; neither is resident credit.
    (
        patches * row * SIZEOF_F32 + projected + 6 * ctx * SIZEOF_I32 + resize,
        projected + 3 * ctx * SIZEOF_I32,
    )
}

// Maximum glm53_vision_smart_resize grid: 8000 merged tokens, four patches
// each. Encoder buffers: patch; a/b/q/k/v/attention; QKV; gate/up/mid.
// The merger reuses these same allocations.
fn glm_media_bytes() -> u64 {
    let rows = 8000 * 4;
    let encoder = rows * (1176 + 6 * 1024 + 3072 + 3 * 4096) * SIZEOF_F32;
    // Host patches live until GPU encoding returns. The bridge retains all
    // four images' host embeddings until sync completes, and encodes them
    // before validating their spans against the prompt, so ctx cannot cap
    // this allocation. Decode/resize finishes before the larger GPU peak.
    let host_patches = rows * 1176 * SIZEOF_F32;
    let embeddings = 4 * 8000 * 4096 * SIZEOF_F32;
    encoder + host_patches + embeddings
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
    let rows = ctx.min(STEP_MEDIA_ROWS);
    // 728^2/169 and 504^2/81 both equal 56^2 pixels per image token.
    // Prepared crops remain live through native encoding and prefill.
    let crops = rows * STEP_PIXELS_PER_TOKEN * SIZEOF_F32;
    let native = workspace + rows * hidden * SIZEOF_F32;
    // Decoder/EXIF copies and RGB horizontal resize can hold two 128 MiB
    // buffers. Two maximum base RGB images also bound crop/resize staging
    // and filter coefficients; F32 crop normalization fits below this peak.
    let prepare = 2 * STEP_RGB_LIMIT + 2 * STEP_SOURCE_EDGE.pow(2) * 3;
    // Session graph initialization allocates native vision buffers even on
    // text-only requests and retains them until ds4_session_free. A later
    // image preparation therefore overlaps that entire native allocation.
    crops + native + prepare
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

// C dots3_graph_memory_estimate without the separately priced serial MTP.
// Each persistent bank owns its complete prefill scratch.
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

// C dots3_spec_bytes: one scalar draft, target hidden rows and four-row undo.
// Predictor layer46 is local attention and has no DSA score workspace.
fn dots3_mtp_bytes(s: Shape, ctx: u64, native: u32) -> u64 {
    let rot = u64::from(s.n_rot);
    let local = u64::from(s.n_swa_kv_lora);
    let q_lora = u64::from(s.n_lora_q);
    let index = u64::from(s.n_indexer_head_dim);
    let ring = ctx.min(u64::from(s.n_swa) + 1);
    let mut cache = 0;
    let mut norms = 0;
    let mut journal = 0;
    for il in 0..s.n_layer {
        let full = il == 0 || (s.n_swa_period != 0 && il % s.n_swa_period == 1);
        let latent = if full { u64::from(s.n_kv_lora) } else { local };
        let row = (latent + rot) * SIZEOF_U16 + if full { index * SIZEOF_F32 } else { 0 };
        journal += DOTS3_TRIAL_ROWS * row;
        if il < s.n_layer.saturating_sub(s.n_nextn_predict) {
            cache += if full { ctx } else { ring } * row;
            norms += q_lora + latent + rot + if full { 2 * index } else { 0 };
        }
    }
    let scratch = dots3_graph_bytes(s, ctx, 1) - cache - (norms + ctx) * SIZEOF_F32;
    scratch
        + ring * (local + rot) * SIZEOF_U16
        + (q_lora + local + rot) * SIZEOF_F32
        + (u64::from(native).max(DOTS3_TRIAL_ROWS) + 3) * u64::from(s.n_embd) * SIZEOF_F32
        + journal
        + (DOTS3_TRIAL_ROWS + 1) * u64::from(s.n_vocab) * SIZEOF_F32
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
    // Media stays lazy until image use. Checkpoint slabs reserve virtual
    // addresses at fit but map physical pages only on capture. Both costs
    // must remain available after a successful fit, without resident credit.
    let banks = facts.banks_fitted.unwrap_or(1);
    facts
        .per_bank_bytes
        .unwrap_or(0)
        .saturating_mul(u64::from(banks))
        .saturating_add(facts.mtp_state_bytes.unwrap_or(0))
        .saturating_add(facts.scratch_bytes.unwrap_or(0))
        .saturating_add(facts.ple_bytes.unwrap_or(0))
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

#[cfg(any(target_os = "macos", test))]
fn parse_vm_stat(text: &str) -> Option<u64> {
    let page_size = text
        .lines()
        .next()?
        .split_once("page size of ")?
        .1
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    if page_size == 0 {
        return None;
    }
    let mut pages = 0u64;
    // vm_stat subtracts speculative pages from its printed free count.
    // Purgeable pages overlap other queues, so do not add them again.
    for key in ["Pages free:", "Pages inactive:", "Pages speculative:"] {
        let value = text.lines().find_map(|line| line.strip_prefix(key))?;
        let count = value.trim().trim_end_matches('.').parse::<u64>().ok()?;
        pages = pages.checked_add(count)?;
    }
    pages.checked_mul(page_size)
}

#[cfg(target_os = "macos")]
fn meminfo_available() -> u64 {
    let Ok(out) = std::process::Command::new("/usr/bin/vm_stat").output() else {
        return 0;
    };
    if !out.status.success() {
        return 0;
    }
    std::str::from_utf8(&out.stdout)
        .ok()
        .and_then(parse_vm_stat)
        .unwrap_or(0)
}

#[cfg(not(target_os = "macos"))]
fn meminfo_available() -> u64 {
    meminfo_kb("MemAvailable:")
}

#[cfg(not(target_os = "macos"))]
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
        SHAPE_K2_HORIZON_375B, SHAPE_KEXAONE_236B, SHAPE_LING30_FLASH_VL, SHAPE_MOTIF3,
        SHAPE_QWEN38_FLASH_NEXT, SHAPE_SOLAR_OPEN2_250B, SHAPE_STEP37_FLASH,
    };
    use std::io::Write;

    // The native log for `-c 8192` prints KV=0.06 GiB state=76.6 MiB
    // workspace=1.24 GiB per graph; the quote has to reach the same bytes or
    // --check-config approves a configuration the allocator cannot fund.
    #[test]
    fn ling_quote_prices_the_graph_and_the_checkpoint_pool() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(LING_PREFILL_CHUNK_ENV);
        let _partial = EnvGuard::unset("DS4_SERVER_FORK_PARTIAL");
        let s = SHAPE_LING30_FLASH_VL;
        assert_eq!(ling_latent_bytes(s, 8192), 66_060_288);
        assert_eq!(ling_state_bytes(s), 80_281_600);
        // C ling3vl_graph_alloc control_pool: 35 KDA layers, not in the
        // checkpoint slab (that slab is state_bytes only).
        assert_eq!(ling_control_bytes(s), 7_477_120);
        assert_eq!(ling_graph_bytes(s, LING_NATIVE_DEFAULT), 1_336_815_616);
        assert_eq!(ling_checkpoint_pool_bytes(s), 80_281_600 * CHECKPOINT_SLOTS);

        let mut facts = EngineFacts::default();
        let req = ServingRequest {
            ctx: 8192,
            max_seqs: MaxSeqs::Fixed(2),
            ..ServingRequest::default()
        };
        let caps = serving_caps(ModelFamily::Ling3Vl, Variant::Ling30FlashVl);
        fill_quote_facts(
            &mut facts,
            &req,
            caps,
            Some(s),
            QuoteHost {
                weights_bytes: 78 * GIB,
                mtp_bytes: 0,
                available_bytes: 110 * GIB,
                native_chunk: None,
                vision: true,
            },
        );
        assert_eq!(facts.per_bank_bytes, Some(1_483_157_504));
        assert_eq!(facts.checkpoint_pool_bytes, Some(2_569_011_200));
        // No predictor block, and the graph is per bank rather than shared.
        assert_eq!(facts.mtp_state_bytes, Some(0));
        assert_eq!(facts.scratch_bytes, Some(0));
        // Banks stay live; the first image allocates an independent
        // ling3vl_graph_alloc. C ling3vl_session_bytes is that language
        // memory plus the 16,384-patch ViT workspace and projected rows.
        assert_eq!(facts.media_reserve_bytes, Some(2_395_025_408));

        let mut serial = EngineFacts::default();
        let serial_req = ServingRequest {
            ctx: 8192,
            max_seqs: MaxSeqs::Fixed(1),
            ..ServingRequest::default()
        };
        fill_quote_facts(
            &mut serial,
            &serial_req,
            caps,
            Some(s),
            QuoteHost {
                weights_bytes: 78 * GIB,
                mtp_bytes: 0,
                available_bytes: 110 * GIB,
                native_chunk: None,
                vision: true,
            },
        );
        // Width 1 is the serial session itself; do not price a second graph.
        assert_eq!(serial.media_reserve_bytes, Some(911_867_904));
    }

    #[test]
    fn ling_native_chunk_is_published() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(LING_PREFILL_CHUNK_ENV);
        let req = ServingRequest {
            ctx: 8192,
            native_chunk: Some(1024),
            ..ServingRequest::default()
        };
        let caps = serving_caps(ModelFamily::Ling3Vl, Variant::Ling30FlashVl);
        let mut facts = EngineFacts::default();
        fill_quote_facts(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_LING30_FLASH_VL),
            QuoteHost {
                weights_bytes: 78 * GIB,
                mtp_bytes: 0,
                available_bytes: 110 * GIB,
                native_chunk: None,
                vision: false,
            },
        );
        assert_eq!(facts.native_chunk, Some(1024));
        let p = resolve_plan(&req, Some(caps), &facts);
        assert!(p
            .env_overrides()
            .contains(&(LING_PREFILL_CHUNK_ENV.into(), "1024".into())));
    }

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
        assert!(facts.media_reserve_bytes.unwrap() > 0);
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
        let req = ServingRequest {
            backend: Backend::Cpu,
            ..ServingRequest::default()
        };
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
            None,
            false,
            false,
        );
        assert_eq!(facts.shared_weights_bytes, Some(140));
        assert!(facts.host_available_bytes.unwrap() > 0);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(plan.quote.is_some(), "{:?}", plan.to_json());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Minimal GGUF directory: names + dims only, one F32 tensor per entry.
    fn write_slice_gguf(path: &Path, tensors: &[(&str, u64)]) {
        fn put_u32(buf: &mut Vec<u8>, v: u32) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        fn put_u64(buf: &mut Vec<u8>, v: u64) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        fn put_str(buf: &mut Vec<u8>, s: &str) {
            put_u64(buf, s.len() as u64);
            buf.extend_from_slice(s.as_bytes());
        }

        const GGUF_MAGIC: u32 = 0x4655_4747;
        const GGUF_VERSION: u32 = 3;
        const GGUF_TYPE_F32: u32 = 0;
        const ALIGN: u64 = 32;

        let mut buf = Vec::new();
        put_u32(&mut buf, GGUF_MAGIC);
        put_u32(&mut buf, GGUF_VERSION);
        put_u64(&mut buf, tensors.len() as u64);
        put_u64(&mut buf, 0);
        let mut rel = 0u64;
        for (name, elems) in tensors {
            put_str(&mut buf, name);
            put_u32(&mut buf, 1);
            put_u64(&mut buf, *elems);
            put_u32(&mut buf, GGUF_TYPE_F32);
            put_u64(&mut buf, rel);
            rel += elems * SIZEOF_F32;
        }
        let data_pos = (buf.len() as u64).div_ceil(ALIGN) * ALIGN;
        buf.resize((data_pos + rel) as usize, 0);
        std::fs::write(path, buf).unwrap();
    }

    #[test]
    fn worker_slice_prices_only_its_layers() {
        let _env = lock_test_env();
        let _man = EnvGuard::unset(WEIGHT_IPC_MANIFEST_ENV);
        let _scope = EnvGuard::unset(WEIGHT_IPC_SCOPE_ENV);
        let dir = std::env::temp_dir().join(format!("ds4-slice-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sliced.gguf");
        // 4 blocks of 64 F32 each, plus embedding and output heads.
        write_slice_gguf(
            &path,
            &[
                ("token_embd.weight", 128),
                ("blk.0.attn_norm.weight", 64),
                ("blk.1.attn_norm.weight", 64),
                ("blk.2.attn_norm.weight", 64),
                ("blk.3.attn_norm.weight", 64),
                ("output_norm.weight", 32),
                ("output.weight", 96),
                ("hc_input.norm.weight", 16),
                // Groups a sliced map never retains, whoever owns the head.
                ("token_embd_mtp.weight", 256),
                ("vblk.0.attn_norm.weight", 512),
                ("mtp.0.attn_norm.weight", 512),
            ],
        );
        let whole = gguf_span_bytes(&path, 1);

        // Middle worker: two blocks, no embedding, no head.
        let middle = WeightSlice {
            start: 1,
            end: 2,
            output: false,
        };
        assert_eq!(gguf_slice_span_bytes(&path, 1, middle), 2 * 64 * SIZEOF_F32);

        // Tail worker owns the output group: the native allowlist only, so
        // the vision/MTP groups stay off the price. `u32::MAX` runs to the
        // last block.
        let tail = WeightSlice {
            start: 3,
            end: u32::MAX,
            output: true,
        };
        assert_eq!(
            gguf_slice_span_bytes(&path, 1, tail),
            (64 + 32 + 96 + 16) * SIZEOF_F32
        );

        // Head worker pulls the token embedding in with layer 0, but not the
        // sidecar embedding that shares its prefix.
        let head = WeightSlice {
            start: 0,
            end: 0,
            output: false,
        };
        assert_eq!(
            gguf_slice_span_bytes(&path, 1, head),
            (128 + 64) * SIZEOF_F32
        );

        let mut facts = EngineFacts::default();
        let req = ServingRequest {
            backend: Backend::Cpu,
            ..ServingRequest::default()
        };
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        attach_host_quote(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            Some(&path),
            None,
            None,
            None,
            1,
            Some(middle),
            false,
            false,
        );
        assert_eq!(facts.shared_weights_bytes, Some(2 * 64 * SIZEOF_F32));
        assert!(facts.shared_weights_bytes.unwrap() < whole);

        // Undistributed serving still prices the whole artifact.
        let mut full = EngineFacts::default();
        attach_host_quote(
            &mut full,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            Some(&path),
            None,
            None,
            None,
            1,
            None,
            false,
            false,
        );
        assert_eq!(full.shared_weights_bytes, Some(whole));
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
            assert_eq!(facts.shared_weights_bytes, Some(193));
            assert!(facts.ipc_pending);
        }

        let facts = attach_ipc(&a, Some(&mtp), Some(&vision), Some(&dspark), false);
        assert_eq!(facts.shared_weights_bytes, Some(193));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_available_bytes_reads_this_host() {
        let _env = lock_test_env();
        assert!(host_available_bytes(Backend::Cpu) > 0);
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
    fn native_cli_overrides_default() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut req = ServingRequest::default();
        req.native_chunk = Some(512);
        req.sched_chunk = Some(512);
        let facts = fill_qwen(&req, qwen_host(None));
        assert_eq!(facts.native_chunk, Some(512));
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let p = resolve_plan(&req, Some(caps), &facts);
        assert!(p.may_listen(), "{:?}", p.issues);
        assert!(p
            .env_overrides()
            .contains(&(QWEN_PREFILL_CHUNK_ENV.into(), "512".into())));
        let _small_env = EnvGuard::set(QWEN_PREFILL_CHUNK_ENV, "128");
        assert_eq!(fill_qwen(&req, qwen_host(None)).native_chunk, Some(512));
        req.ctx = 32768;
        req.native_chunk = Some(65536);
        assert_eq!(
            fill_qwen(&req, qwen_host(None)).native_chunk,
            Some(QWEN_NATIVE_MAX)
        );
        req.ctx = 128;
        assert_eq!(fill_qwen(&req, qwen_host(None)).native_chunk, Some(128));
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
                facts
                    .media_per_extra_bank_bytes
                    .unwrap_or(0)
                    .saturating_mul(u64::from(banks.saturating_sub(1))),
            )
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
            per_bank + facts.media_per_extra_bank_bytes.unwrap()
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
    fn inkling_embedded_media_reserve() {
        let _env = lock_test_env();
        let _floor = EnvGuard::set(SESSION_FIT_ENV, "0");
        let caps = serving_caps(ModelFamily::Inkling, Variant::InklingSmall);
        for (ctx, expected) in [
            (1024, 307_757_056),
            (8192, 583_008_256),
            (16384, 899_809_280),
            (32768, 1_797_521_408),
            (65536, 1_797_783_552),
        ] {
            let req = ServingRequest {
                ctx,
                mem_floor_gb: 0,
                ..ServingRequest::default()
            };
            let mut facts = fill_family(
                ModelFamily::Inkling,
                Variant::InklingSmall,
                SHAPE_INKLING_SMALL,
                &req,
                qwen_host(None),
            );
            assert!(!caps.media_serial);
            assert_eq!(facts.media_reserve_bytes, Some(expected));
            facts.host_available_bytes = Some(facts_cost(&facts, &req, 1) - 1);
            assert!(resolve_plan(&req, Some(caps), &facts)
                .issues
                .iter()
                .any(|issue| issue.code == "quote_overflow"));
            let resident = resident_runtime(&facts);
            facts.media_reserve_bytes = Some(0);
            assert_eq!(resident_runtime(&facts), resident);
        }
    }

    #[test]
    fn qwen_auto_shrinks_media_cost() {
        let _env = lock_test_env();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let mut req = ServingRequest {
            ctx: 65536,
            ..ServingRequest::default()
        };
        let mut facts = fill_qwen(&req, qwen_host(None));
        let serial_media = 14_684_782_592;
        let one_bank =
            facts_cost(&facts, &req, 1) - facts.media_reserve_bytes.unwrap() + serial_media;
        facts.host_available_bytes = Some(one_bank);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(plan.may_listen(), "{:?}", plan.issues);
        assert_eq!(plan.effective.max_seqs, 1);
        assert_eq!(plan.quote.unwrap().media_reserve, serial_media);
        assert_eq!(plan.quote.unwrap().total, one_bank);
        req.max_seqs = MaxSeqs::Fixed(2);
        assert!(resolve_plan(&req, Some(caps), &facts).has_errors());
    }

    #[test]
    fn qwen_embedded_vision_reserve() {
        let _env = lock_test_env();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        // Native vision charge plus the bounded decoder/resize peak. Width
        // two also retains one bank's projected features and M-RoPE rows.
        for (ctx, serial, two_banks) in [
            (1024, 438_984_704, 449_482_752),
            (16384, 3_830_841_344, 3_998_810_112),
            (65536, 14_684_782_592, 15_356_657_664),
            (262144, 14_689_501_184, 15_363_735_552),
        ] {
            for (max_seqs, expected) in [(MaxSeqs::Off, serial), (MaxSeqs::Auto, two_banks)] {
                let req = ServingRequest {
                    ctx,
                    max_seqs,
                    ..ServingRequest::default()
                };
                let mut facts = fill_qwen(&req, qwen_host(None));
                assert!(!caps.media_serial);
                assert_eq!(facts.media_reserve_bytes, Some(serial));
                facts.host_available_bytes = Some(facts_cost(&facts, &req, 2));
                assert_eq!(
                    resolve_plan(&req, Some(caps), &facts)
                        .quote
                        .unwrap()
                        .media_reserve,
                    expected
                );
                facts.host_available_bytes = Some(facts_cost(&facts, &req, 1) - 1);
                assert!(resolve_plan(&req, Some(caps), &facts).has_errors());
                let resident = resident_runtime(&facts);
                facts.media_reserve_bytes = Some(0);
                facts.media_per_extra_bank_bytes = Some(0);
                assert_eq!(resident_runtime(&facts), resident);
            }
        }
    }

    #[test]
    fn cpu_prefill_reserves_full_rows() {
        let _env = lock_test_env();
        for shape in [crate::shape::SHAPE_FLASH, crate::shape::SHAPE_PRO] {
            for ctx in [8192, 262144] {
                let req = ServingRequest {
                    backend: Backend::Cpu,
                    ctx,
                    native_chunk: Some(256),
                    ..ServingRequest::default()
                };
                let mut facts =
                    fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
                // All three outer HC buffers and both full-prompt Q/heads
                // coexist with the session's decode scratch during prefill.
                let (_, decode) = deepseek_cpu_bytes(shape, ctx as u64);
                let minimum = decode
                    + ctx as u64
                        * SIZEOF_F32
                        * (3 * shape.n_hc as u64 * shape.n_embd as u64
                            + 2 * shape.n_head as u64 * shape.n_head_dim as u64);
                assert!(facts.scratch_bytes.unwrap() >= minimum);
                facts.host_available_bytes = Some(facts_cost(&facts, &req, 1) - 1);
                let plan = resolve_plan(
                    &req,
                    Some(serving_caps(shape.family, shape.variant)),
                    &facts,
                );
                assert!(plan
                    .issues
                    .iter()
                    .any(|issue| issue.code == "quote_overflow"));
            }
        }
    }

    #[test]
    fn missing_cuda_memory_is_unknown() {
        assert_eq!(quote_ceiling(None), 0);
        assert_eq!(quote_device(50 * GIB, None), 0);
    }

    #[test]
    fn cpu_deepseek_uses_cpu_buffers() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(SOLAR_PREFILL_CHUNK_ENV);
        for (shape, ctx, cache, scratch) in [
            (crate::shape::SHAPE_FLASH, 8192, 136_389_632, 1_591_858),
            (crate::shape::SHAPE_PRO, 8192, 196_331_520, 2_405_186),
            (crate::shape::SHAPE_FLASH, 262144, 3_630_769_152, 2_163_250),
            (crate::shape::SHAPE_PRO, 262144, 5_198_170_112, 2_976_578),
        ] {
            for native in [256, 4096] {
                let req = ServingRequest {
                    backend: Backend::Cpu,
                    ctx,
                    native_chunk: Some(native),
                    ..ServingRequest::default()
                };
                let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
                assert_eq!(facts.per_bank_bytes, Some(cache));
                assert_eq!(deepseek_cpu_bytes(shape, ctx as u64), (cache, scratch));
                assert!(facts.scratch_bytes.unwrap() > scratch);
                assert_eq!(facts.mtp_state_bytes, Some(0));
            }
        }
    }

    #[test]
    fn equal_ram_vram_is_discrete() {
        assert_eq!(quote_ceiling(Some((24 * GIB, GIB))), GIB);
        assert_eq!(quote_ceiling(Some((48 * GIB, GIB))), GIB);
    }

    #[test]
    fn native_quote_glm_vision() {
        let _env = lock_test_env();
        let req = ServingRequest {
            ctx: 2048,
            ..ServingRequest::default()
        };
        let caps = serving_caps(ModelFamily::Glm53, Variant::Glm53Flash);
        for (host_vision, loaded) in [(true, false), (false, true), (false, false)] {
            let mut facts = EngineFacts {
                vision_loaded: loaded,
                ..EngineFacts::default()
            };
            fill_quote_facts(
                &mut facts,
                &req,
                caps,
                Some(SHAPE_GLM53_FLASH),
                QuoteHost {
                    vision: host_vision,
                    ..qwen_host(None)
                },
            );
            assert_eq!(
                facts.media_reserve_bytes,
                Some(if host_vision || loaded {
                    3_577_856_000
                } else {
                    0
                })
            );
        }
    }

    #[test]
    fn native_quote_solar_workspace() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(SOLAR_PREFILL_CHUNK_ENV);
        let _group = EnvGuard::unset("DS4_CUDA_SOLAR_GQA_GROUPED");
        let _split = EnvGuard::unset("DS4_CUDA_SOLAR_GQA_CHUNK");
        for (ctx, native, scratch, bank_extra) in [
            (8192, 2048, 1_784_901_696, 5_046_280),
            (262144, 2048, 1_916_956_736, 137_101_320),
            (8192, 256, 241_538_112, 5_046_280),
            (32, 32, 37_575_360, 819_720),
        ] {
            let req = ServingRequest {
                ctx,
                native_chunk: Some(native),
                ..ServingRequest::default()
            };
            let shape = SHAPE_SOLAR_OPEN2_250B;
            let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
            assert_eq!(facts.scratch_bytes, Some(scratch));
            assert_eq!(
                facts.per_bank_bytes,
                Some(bank_kv_bytes(shape, ctx as u64, native) + bank_extra)
            );
        }
    }

    #[test]
    fn native_quote_exaone_workspace() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(EXAONE_PREFILL_CHUNK_ENV);
        for (shape, shared, bank) in [
            (SHAPE_KEXAONE_236B, 428_113_920, 499_089_984),
            (SHAPE_K2_HORIZON_375B, 786_235_392, 2_049_593_152),
        ] {
            let mut req = ServingRequest::default();
            let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
            assert_eq!(facts.scratch_bytes, Some(shared));
            assert_eq!(facts.per_bank_bytes, Some(bank));
            req.max_seqs = MaxSeqs::Off;
            let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
            assert_eq!(
                facts.per_bank_bytes,
                Some(bank - u64::from(shape.n_vocab) * SIZEOF_F32)
            );
        }
    }

    #[test]
    fn fitted_media_stays_reserved() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::unset(STEP_PREFILL_CHUNK_ENV);
        let _batch = EnvGuard::set(FIT_HEADROOM_ENV, "0");
        let _session = EnvGuard::set(SESSION_HEADROOM_ENV, "0");
        let req = ServingRequest {
            ctx: 8192,
            mem_floor_gb: 0,
            max_seqs: MaxSeqs::Fixed(2),
            mtp_mode: MtpMode::Off,
            ..ServingRequest::default()
        };
        let caps = serving_caps(ModelFamily::Step37, Variant::Step37Flash);
        let mapped = 10 * GIB;
        let media = 1_037_997_440;
        for (live, can_listen) in [(media - 1, false), (media, true)] {
            let mut facts = EngineFacts {
                banks_fitted: Some(2),
                cont_lane: Some(true),
                vision_loaded: true,
                ..EngineFacts::default()
            };
            fill_quote_facts(
                &mut facts,
                &req,
                caps,
                Some(SHAPE_STEP37_FLASH),
                QuoteHost {
                    weights_bytes: mapped,
                    available_bytes: live,
                    vision: true,
                    ..qwen_host(None)
                },
            );
            assert_eq!(facts.media_reserve_bytes, Some(media));
            let live = live + facts.checkpoint_pool_bytes.unwrap();
            credit_resident(&mut facts, live, mapped);
            let plan = resolve_plan(&req, Some(caps), &facts);
            assert_eq!(plan.may_listen(), can_listen, "{:?}", plan.issues);
        }
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
    fn checkpoint_pages_stay_reserved() {
        let facts = EngineFacts {
            banks_fitted: Some(2),
            cont_lane: Some(true),
            per_bank_bytes: Some(3 * GIB),
            scratch_bytes: Some(GIB),
            checkpoint_pool_bytes: Some(2 * GIB),
            ..EngineFacts::default()
        };
        assert_eq!(resident_runtime(&facts), 7 * GIB);
        let req = ServingRequest {
            mem_floor_gb: 0,
            max_seqs: MaxSeqs::Fixed(2),
            ..ServingRequest::default()
        };
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        for (live, allowed) in [(2 * GIB - 1, false), (2 * GIB, true)] {
            let mut facts = facts.clone();
            credit_resident(&mut facts, live, 0);
            let p = resolve_plan(&req, Some(caps), &facts);
            assert_eq!(p.may_listen(), allowed, "{:?}", p.issues);
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
        assert!(resolve_plan(&req, Some(caps), &facts).has_errors());
        // Open banks leave media and checkpoint pages lazy. Supply these
        // reserves in live memory before checking resident runtime credit.
        let with_media = leftover
            + facts.media_reserve_bytes.unwrap()
            + facts.media_per_extra_bank_bytes.unwrap()
            + facts.checkpoint_pool_bytes.unwrap();
        credit_resident(&mut facts, with_media, mapped);
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
        let avail = 50 * GIB;
        let vram = 24 * GIB;
        let free = 20 * GIB;
        assert_eq!(quote_ceiling(Some((vram, free))), free);
        assert_eq!(quote_ceiling(None), 0);
        assert_eq!(
            quote_device(
                avail,
                Some(crate::serving_cuda::Device {
                    uuid: String::new(),
                    integrated: true,
                })
            ),
            avail
        );
        assert_eq!(parse_nvidia_csv("[N/A], [N/A]\n"), None);
        assert_eq!(
            parse_nvidia_csv("24576, 20480\n"),
            Some((24576 * MIB, 20480 * MIB))
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn smi_uses_first_visible_gpu() {
        if let Ok(expected) = std::env::var("DS4_TEST_CUDA_FREE") {
            assert_eq!(
                host_available_bytes(Backend::Cuda),
                expected.parse::<u64>().unwrap()
            );
            return;
        }
        let dir = std::env::temp_dir().join(format!("ds4-driver-smi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("cuda.c");
        std::fs::write(&source, include_str!("../tests/fixtures/cuda_probe.c")).unwrap();
        let build = std::process::Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&source)
            .arg("-o")
            .arg(dir.join("libcuda.so.1"))
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let smi = dir.join("nvidia-smi");
        std::fs::write(&smi, r#"#!/bin/sh
if [ "$DS4_TEST_SMI_FAIL" = "1" ]; then exit 1; fi
for arg in "$@"; do
    case "$arg" in
        --id=0|--id=GPU-00000000-0000-0000-0000-000000000000) printf '24576, 20000\n'; exit 0 ;;
        --id=1|--id=GPU-11111111-1111-1111-1111-111111111111) printf '8192, 2048\n'; exit 0 ;;
        --query-gpu=uuid) printf 'GPU-00000000-0000-0000-0000-000000000000\nGPU-11111111-1111-1111-1111-111111111111\n'; exit 0 ;;
    esac
done
exit 1
"#).unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&smi, std::fs::Permissions::from_mode(0o755)).unwrap();
        for (visible, order, free) in [
            ("0", "FASTEST_FIRST", 2048),
            ("0", "PCI_BUS_ID", 20000),
            ("1", "FASTEST_FIRST", 20000),
            ("1,0", "PCI_BUS_ID", 2048),
            (
                "GPU-11111111-1111-1111-1111-111111111111",
                "FASTEST_FIRST",
                2048,
            ),
            ("GPU-11111111", "FASTEST_FIRST", 2048),
            ("0", "FASTEST_FIRST", 0),
        ] {
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "serving_host::tests::smi_uses_first_visible_gpu",
                    "--nocapture",
                ])
                .env("LD_LIBRARY_PATH", &dir)
                .env(
                    "PATH",
                    format!(
                        "{}:{}",
                        dir.display(),
                        std::env::var("PATH").unwrap_or_default()
                    ),
                )
                .env("CUDA_VISIBLE_DEVICES", visible)
                .env("CUDA_DEVICE_ORDER", order)
                .env("DS4_TEST_CUDA_FREE", (free * MIB).to_string())
                .env("DS4_TEST_SMI_FAIL", if free == 0 { "1" } else { "0" })
                .output()
                .unwrap();
            assert!(
                child.status.success(),
                "{visible} {order}: {} {}",
                String::from_utf8_lossy(&child.stdout),
                String::from_utf8_lossy(&child.stderr)
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
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
    fn deepseek_short_page_quote() {
        let _env = lock_test_env();
        let _raw = EnvGuard::unset("DS4_METAL_GRAPH_RAW_CAP");
        let _vmm = EnvGuard::unset("DS4_BATCH_VMM_COMP");
        let _poison = EnvGuard::unset("DS4_BATCH_SLAB_POISON");
        let _fp8 = EnvGuard::unset("DS4_CUDA_FP8_KV");
        let _fp4 = EnvGuard::unset("DS4_CUDA_FP4_INDEX");
        let _band = EnvGuard::unset("DS4_CONT_ADMIT_BAND_X1024");
        let shape = crate::shape::SHAPE_FLASH;
        let req = ServingRequest {
            ctx: 2048,
            max_seqs: MaxSeqs::Fixed(2),
            native_chunk: Some(64),
            mtp_mode: MtpMode::Off,
            ..ServingRequest::default()
        };
        let facts = fill_family(shape.family, shape.variant, shape, &req, qwen_host(None));
        // 62 separate packed slabs round to 124 MiB total, shared by both
        // banks. Banded admission costs 132689920 B, versus 73826304 B
        // of unrounded F32+packed cache capacity in the old two-bank quote.
        let extra = (132_689_920u64 - 73_826_304) / 2;
        assert_eq!(facts.per_bank_bytes, Some(120_972_952 + extra));
        assert_eq!(
            deepseek_page_extra(shape, 2048, 1),
            132_689_920 - 36_913_152
        );
        assert_eq!(deepseek_page_extra(shape, 2048, 4), 0);
        {
            let _fp8 = EnvGuard::set("DS4_CUDA_FP8_KV", "off");
            // F32 primary's 21 ratio-4 slabs need two pages each at width2.
            assert_eq!(
                deepseek_page_extra(shape, 2048, 2),
                177_633_280 - 73_826_304
            );
        }
        {
            let _vmm = EnvGuard::set("DS4_BATCH_VMM_COMP", "0");
            assert_eq!(deepseek_page_extra(shape, 2048, 2), 0);
        }
        let mut fitted = EngineFacts {
            banks_fitted: Some(1),
            ..EngineFacts::default()
        };
        fill_quote_facts(
            &mut fitted,
            &req,
            serving_caps(shape.family, shape.variant),
            Some(shape),
            qwen_host(None),
        );
        assert_eq!(
            fitted.per_bank_bytes,
            Some(120_972_952 + 132_689_920 - 36_913_152)
        );
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
    fn dots3_mtp_quote_matches_native() {
        let _env = lock_test_env();
        for (ctx, native, expected) in [
            (4096, 128, 26_068_040),
            (1024, 32, 24_101_960),
            (262144, 4096, 107_332_680),
        ] {
            for mode in [MtpMode::Off, MtpMode::Auto, MtpMode::On] {
                let req = ServingRequest {
                    ctx,
                    native_chunk: Some(native),
                    mtp_mode: mode,
                    ..ServingRequest::default()
                };
                let facts = fill_family(
                    ModelFamily::Dots3Note,
                    Variant::Dots3NotePrev,
                    SHAPE_DOTS3_NOTE_PREV,
                    &req,
                    qwen_host(Some(native)),
                );
                assert_eq!(
                    facts.mtp_state_bytes,
                    Some(if mode == MtpMode::On { expected } else { 0 }),
                    "ctx={ctx}, mode={mode:?}"
                );
            }
        }
    }

    #[test]
    fn dots3_bank_quote_has_local_pool() {
        let _env = lock_test_env();
        let _chunk = EnvGuard::set(DOTS3_PREFILL_CHUNK_ENV, "64");
        for (reuse, expected) in [
            (PrefixReuse::Partial, 1_178_800_128),
            (PrefixReuse::Auto, 0),
            (PrefixReuse::Exact, 0),
        ] {
            let req = ServingRequest {
                ctx: 2048,
                max_seqs: MaxSeqs::Fixed(2),
                prefix_reuse: reuse,
                mtp_mode: MtpMode::Off,
                ..ServingRequest::default()
            };
            let facts = fill_family(
                ModelFamily::Dots3Note,
                Variant::Dots3NotePrev,
                SHAPE_DOTS3_NOTE_PREV,
                &req,
                qwen_host(Some(64)),
            );
            assert_eq!(facts.checkpoint_pool_bytes, Some(expected));
            assert_eq!(
                facts.per_bank_bytes,
                Some(
                    dots3_graph_bytes(SHAPE_DOTS3_NOTE_PREV, 2048, 64)
                        + u64::from(SHAPE_DOTS3_NOTE_PREV.n_vocab) * SIZEOF_F32
                )
            );
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
    fn step_media_retained_buffers() {
        let _env = lock_test_env();
        let mut req = ServingRequest::default();
        for (ctx, expected) in [
            (1024, 650_810_752),
            (8192, 1_037_997_440),
            (16384, 1_037_997_440),
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
    fn exaone_checkpoint_uses_lllg() {
        let _env = lock_test_env();
        let _partial = EnvGuard::unset("DS4_SERVER_FORK_PARTIAL");
        for (reuse, ctx, expected) in [
            (PrefixReuse::Partial, 512, 603_979_776),
            (PrefixReuse::Partial, 64, 301_989_888),
            (PrefixReuse::Auto, 512, 0),
            (PrefixReuse::Exact, 512, 0),
            (PrefixReuse::Off, 512, 0),
        ] {
            let req = ServingRequest {
                prefix_reuse: reuse,
                ctx,
                ..ServingRequest::default()
            };
            let facts = fill_family(
                ModelFamily::ExaoneMoe,
                Variant::Kexaone236B,
                SHAPE_KEXAONE_236B,
                &req,
                QuoteHost {
                    weights_bytes: GIB,
                    mtp_bytes: 0,
                    available_bytes: 100 * GIB,
                    native_chunk: Some(32),
                    vision: false,
                },
            );
            assert_eq!(facts.checkpoint_pool_bytes, Some(expected));
        }
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
        assert_eq!(
            bank_kv_bytes(SHAPE_KEXAONE_236B, u64::from(ctx), native),
            want
        );
        // The bank also owns its decode workspace and two logits rows.
        assert_eq!(facts.per_bank_bytes, Some(want + 2_064_960));
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
            None,
            vision.is_some(),
            resident,
        );
        facts
    }

    #[test]
    fn ipc_drafter_span_is_shared() {
        let _env = lock_test_env();
        let dir = std::env::temp_dir().join(format!("ds4-quote-drafter-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        let drafter = dir.join("drafter.gguf");
        let manifest = dir.join("weights.manifest");
        std::fs::write(&model, [0u8; 100]).unwrap();
        std::fs::write(&drafter, [0u8; 11]).unwrap();
        let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, manifest.to_str().unwrap());
        let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "both");
        let _disable = EnvGuard::unset("DS4_CUDA_WEIGHT_IPC_NO_DRAFTER");
        for header in ["DS4_WEIGHT_SERVER_IPC_V1", "DS4_WEIGHT_SERVER_VMM_V1"] {
            let record = if header.contains("VMM") {
                "broker /tmp/weights.sock\nalloc 0 drafter 11 0 11 65536".to_owned()
            } else {
                format!("range drafter 11 0 11 {}", "00".repeat(64))
            };
            std::fs::write(&manifest, format!("{header}\n{record}\n")).unwrap();
            let facts = attach_ipc(&model, None, None, Some(&drafter), false);
            assert_eq!(
                facts.shared_weights_bytes,
                Some(gguf_span_bytes(&model, 2) + 11)
            );
            assert_eq!(facts.host_available_bytes, None, "import not yet confirmed");
        }
        for (shared, resident, expected) in [
            (Some(true), true, 0),
            (Some(false), true, 11),
            (None, true, 11),
        ] {
            let mut facts = EngineFacts {
                drafter_shared: shared,
                ..EngineFacts::default()
            };
            let req = ServingRequest::default();
            let caps = serving_caps(ModelFamily::DeepSeek4, Variant::Flash);
            attach_host_quote(
                &mut facts,
                &req,
                caps,
                Some(crate::shape::SHAPE_FLASH),
                Some(&model),
                None,
                None,
                Some(&drafter),
                1,
                None,
                false,
                resident,
            );
            assert_eq!(facts.shared_weights_bytes, Some(expected));
        }
        {
            let _disable = EnvGuard::set("DS4_CUDA_WEIGHT_IPC_NO_DRAFTER", "");
            assert!(!ipc_drafter_planned(11));
        }
        assert!(!ipc_drafter_planned(12), "mismatched artifact size");
        std::fs::write(
            &manifest,
            "DS4_WEIGHT_SERVER_VMM_V1\nalloc 0 base 11 0 11 65536\n",
        )
        .unwrap();
        assert!(!ipc_drafter_planned(11), "no drafter ranges");
        std::fs::write(&manifest, "invalid\nalloc 0 drafter 11 0 11 65536\n").unwrap();
        assert!(!ipc_drafter_planned(11), "invalid manifest header");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn ipc_check_requires_import() {
        let _env = lock_test_env();
        let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/missing/owner.manifest");
        let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "both");
        let req = ServingRequest {
            check_config: true,
            ..ServingRequest::default()
        };
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        let mut facts = EngineFacts::default();
        attach_host_quote(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            None,
            None,
            None,
            None,
            1,
            None,
            false,
            false,
        );
        let p = resolve_plan(&req, Some(caps), &facts);
        assert!(
            !p.may_listen(),
            "unconfirmed import must not pass check-config"
        );
    }

    #[test]
    fn mac_memory_reads_vm_pages() {
        for page_size in [4096, 16384] {
            let text = format!("Mach Virtual Memory Statistics: (page size of {page_size} bytes)\nPages free: 10.\nPages inactive: 20.\nPages speculative: 3.\nPages purgeable: 8.\nPages wired down: 1000.\n");
            assert_eq!(parse_vm_stat(&text), Some(33 * page_size));
        }
        assert_eq!(parse_vm_stat(""), None);
        assert_eq!(
            parse_vm_stat("Mach Virtual Memory Statistics: (page size of 0 bytes)"),
            None
        );
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
            let facts = attach_ipc(&a, Some(&mtp), Some(&vision), Some(&dspark), true);
            assert_eq!(facts.shared_weights_bytes, Some(17 + 11));
        }
        {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "both");
            let facts = attach_ipc(&a, Some(&mtp), None, None, true);
            assert_eq!(facts.shared_weights_bytes, Some(0));
        }
        {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "mtp");
            let facts = attach_ipc(&a, Some(&mtp), None, None, true);
            assert_eq!(facts.shared_weights_bytes, Some(140));
        }
        {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/tmp/ds4-weights.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, "base");
            let facts = attach_ipc(&a, Some(&mtp), None, None, true);
            assert_eq!(facts.shared_weights_bytes, Some(25));
        }

        for scope in ["base", "mtp", "both", "invalid"] {
            let _man = EnvGuard::set(WEIGHT_IPC_MANIFEST_ENV, "/missing/owner.manifest");
            let _scope = EnvGuard::set(WEIGHT_IPC_SCOPE_ENV, scope);
            let facts = attach_ipc(&a, Some(&mtp), None, None, false);
            assert_eq!(facts.shared_weights_bytes, Some(165));
            assert!(facts.ipc_pending);
            assert_eq!(facts.host_available_bytes, None);
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
