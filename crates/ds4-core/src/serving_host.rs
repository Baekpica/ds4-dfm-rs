//! Host adapter that fills serving-plan quote facts without opening an engine.
//!
//! Weights are the mapped GGUF span. Per-bank / scratch / checkpoint / PLE /
//! media are shape quotes. Available memory is the live host observation.

use std::path::{Path, PathBuf};

use crate::gguf::GgufFile;
use crate::serving::{
    EngineFacts, MtpKind, MtpMode, ReuseKind, ServingCaps, ServingRequest, DEFAULT_SCHED_CHUNK,
};
use crate::shape::{ModelFamily, Shape};
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
    let per_bank = shape.map(|s| bank_kv_bytes(s, ctx)).unwrap_or(0);
    let scratch = shape
        .map(|s| {
            u64::from(native)
                .saturating_mul(u64::from(s.n_embd))
                .saturating_mul(u64::from(s.n_layer))
                .saturating_mul(2)
        })
        .unwrap_or(0);
    let checkpoint = if caps.reuse == ReuseKind::Partial {
        per_bank
    } else {
        0
    };
    let mtp_on = req.mtp_mode != MtpMode::Off && caps.mtp != MtpKind::None;
    let mtp_state = if mtp_on {
        shape
            .map(|s| {
                u64::from(s.n_embd)
                    .saturating_mul(u64::from(s.n_layer))
                    .saturating_mul(caps.spec_draft_min.max(1) as u64)
                    .saturating_mul(2)
            })
            .unwrap_or(0)
    } else {
        0
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
/// After `Model::open`, mapped weights have already left MemAvailable.
/// `resident` credits that span back so the quote is not charged twice.
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
    let weights_bytes = model_path
        .map(|path| gguf_span_bytes(path, split_count))
        .unwrap_or(0);
    let mtp_bytes = artifact_span_bytes(mtp_path);
    let vision_bytes = artifact_span_bytes(vision_path);
    let dspark_bytes = artifact_span_bytes(dspark_path);
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
        _ => DEFAULT_SCHED_CHUNK,
    };
    cap.min(ctx)
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
    use crate::serving::{resolve_plan, serving_caps, PREFILL_CHUNK_FENCE};
    use crate::shape::{Variant, SHAPE_QWEN38_FLASH_NEXT};
    use std::io::Write;

    #[test]
    fn fill_quote_facts_names_every_budget() {
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
        assert!(facts.mtp_state_bytes.unwrap() > 0);
        assert!(facts.scratch_bytes.unwrap() > 0);
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
        assert!(quote.mtp_state > 0);
        assert!(quote.scratch > 0);
        assert!(quote.checkpoint_pool > 0);
        assert!(quote.ple > 0);
        assert_eq!(quote.floor, req.mem_floor_gb * GIB);
        assert!(!plan.to_json()["quote"].is_null());
    }

    #[test]
    fn attach_host_quote_reads_the_mapped_span() {
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

        let mut facts = EngineFacts::default();
        let req = ServingRequest::default();
        let caps = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
        attach_host_quote(
            &mut facts,
            &req,
            caps,
            Some(SHAPE_QWEN38_FLASH_NEXT),
            Some(&a),
            Some(&mtp),
            Some(&vision),
            Some(&dspark),
            2,
            true,
            false,
        );
        assert_eq!(facts.shared_weights_bytes, Some(193));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_available_bytes_reads_this_host() {
        assert!(host_available_bytes() > 0);
    }

    #[test]
    fn resident_span_is_credited_to_available() {
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
        let _chunk = EnvGuard::unset(QWEN_PREFILL_CHUNK_ENV);
        let mut req = ServingRequest::default();
        req.native_chunk = Some(8192);
        let facts = fill_qwen(&req, qwen_host(Some(8192)));
        assert_eq!(facts.native_chunk, Some(QWEN_NATIVE_DEFAULT));
    }

    #[test]
    fn qwen_explicit_yield_past_runtime_native_errors() {
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
}
