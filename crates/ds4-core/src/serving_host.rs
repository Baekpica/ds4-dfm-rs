//! Host adapter that fills serving-plan quote facts without opening an engine.
//!
//! Weights are the mapped GGUF span. Per-bank / scratch / checkpoint / PLE /
//! media are shape quotes. Available memory is the live host observation.

use std::path::{Path, PathBuf};

use crate::serving::{
    EngineFacts, MtpKind, MtpMode, ReuseKind, ServingCaps, ServingRequest, DEFAULT_SCHED_CHUNK,
    PREFILL_CHUNK_FENCE,
};
use crate::shape::{ModelFamily, Shape};
use crate::tensors::model_split_sibling_path;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
const DEFAULT_PLE_CACHE_MB: u64 = 2048;

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
    let native = host
        .native_chunk
        .or(facts.native_chunk)
        .or(req.native_chunk)
        .unwrap_or_else(|| family_native_chunk(caps));
    let ctx = req.ctx.max(1) as u64;
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
    facts.host_available_bytes = Some(host.available_bytes);
    if facts.native_chunk.is_none() {
        facts.native_chunk = Some(native.min(PREFILL_CHUNK_FENCE).max(1));
    }
}

/// Live host observation + GGUF span. Used by the CLI pre-open and post-fit.
pub fn attach_host_quote(
    facts: &mut EngineFacts,
    req: &ServingRequest,
    caps: ServingCaps,
    shape: Option<Shape>,
    model_path: Option<&Path>,
    mtp_path: Option<&Path>,
    split_count: u32,
    vision: bool,
) {
    let host = QuoteHost {
        weights_bytes: model_path
            .map(|path| gguf_span_bytes(path, split_count))
            .unwrap_or(0),
        mtp_bytes: file_len(mtp_path),
        available_bytes: host_available_bytes(),
        native_chunk: req.native_chunk,
        vision,
    };
    fill_quote_facts(facts, req, caps, shape, host);
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

fn family_native_chunk(caps: ServingCaps) -> u32 {
    match caps.family {
        ModelFamily::Qwen4Exp => PREFILL_CHUNK_FENCE,
        ModelFamily::Step37 => 4096,
        ModelFamily::Inkling => 1024,
        ModelFamily::Glm53 => 2048,
        _ => DEFAULT_SCHED_CHUNK,
    }
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
    use crate::serving::{resolve_plan, serving_caps};
    use crate::shape::{Variant, SHAPE_QWEN38_FLASH_NEXT};
    use std::io::Write;

    #[test]
    fn fill_quote_facts_names_every_budget() {
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
        assert_eq!(facts.native_chunk, Some(PREFILL_CHUNK_FENCE));

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
            2,
            false,
        );
        assert_eq!(facts.shared_weights_bytes, Some(140));
        assert!(facts.host_available_bytes.unwrap() > 0);
        let plan = resolve_plan(&req, Some(caps), &facts);
        assert!(plan.quote.is_some(), "{:?}", plan.to_json());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_available_bytes_reads_this_host() {
        assert!(host_available_bytes() > 0);
    }
}
