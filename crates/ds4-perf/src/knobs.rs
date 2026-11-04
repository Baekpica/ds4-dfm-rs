/// The Gemma-4 arithmetic surface. The host validates every value strictly
/// at parse time, so an experiment only has to confirm the knob belongs to
/// this family; the range checks here mirror the host so a sweep fails before
/// it spends a model load.
fn validate_gemma4(key: &str, value: &str) -> Option<Result<(), String>> {
    let bad = || {
        Some(Err(format!(
            "unsupported experiment control/value for gemma4_vq: {key}={value}"
        )))
    };
    let number = |min: u32, max: u32| -> Option<Result<(), String>> {
        Some(match value.parse::<u32>() {
            Ok(n) if (min..=max).contains(&n) => Ok(()),
            _ => Err(format!("{key}: expected {min}..{max}, got {value}")),
        })
    };
    match key {
        "DS4_GEMMA4_ATTN_SPLIT_SPAN" => number(16, 4096),
        "DS4_GEMMA4_CPU_MOE_THREADS" => number(1, 64),
        "DS4_GEMMA4_PREFILL_CHUNK" => number(1, 8192),
        "DS4_GEMMA4_VQ_GEMV_VARIANT" => number(0, 2),
        "DS4_GEMMA4_VQ_GEMV_THREADS" => Some(match value.parse::<u32>() {
            Ok(n) if (32..=512).contains(&n) && n % 32 == 0 => Ok(()),
            _ => Err(format!(
                "{key}: expected 32..512 in steps of 32, got {value}"
            )),
        }),
        "DS4_GEMMA4_MOE_MISS" => match value {
            "evict" | "cpu" => Some(Ok(())),
            _ => bad(),
        },
        "DS4_GEMMA4_FULL_KV" => match value {
            "bf16" | "fp8" => Some(Ok(())),
            _ => bad(),
        },
        "DS4_GEMMA4_ATTN_SPLIT_MMA"
        | "DS4_GEMMA4_ATTN_SPLIT"
        | "DS4_GEMMA4_ATTN_FAST"
        | "DS4_GEMMA4_ATTN_PREFILL"
        | "DS4_GEMMA4_ATTN_MMA"
        | "DS4_GEMMA4_FUSE_ROPE"
        | "DS4_GEMMA4_HEAD_GEMV"
        | "DS4_GEMMA4_ROUTER_DOT"
        | "DS4_GEMMA4_ROUTER_PAR"
        | "DS4_GEMMA4_Q8_BATCH"
        | "DS4_GEMMA4_MOE_PERSIST"
        | "DS4_GEMMA4_MOE_FUSED"
        | "DS4_GEMMA4_VQ_PREFILL"
        | "DS4_GEMMA4_EXACT_PREFILL"
        | "DS4_GEMMA4_Q4_DECODE"
        | "DS4_GEMMA4_CPU_MOE"
        | "DS4_GEMMA4_MOE_HOST"
        | "DS4_GEMMA4_VQ_GEMV" => match value {
            "0" | "1" | "true" | "false" | "on" | "off" => Some(Ok(())),
            _ => bad(),
        },
        _ => None,
    }
}

// Only controls whose values are consumed by the native execution path may
// vary inside a matched experiment. Memory/residency identity stays fixed.
pub fn tunable(key: &str) -> bool {
    if key.starts_with("DS4_GEMMA4_") {
        return validate_gemma4(key, "1").is_some();
    }
    matches!(
        key,
        "DS4_QWEN_PREFILL_CHUNK"
            | "DS4_IQUEST_PREFILL_CHUNK"
            | "DS4_QWEN_PLE_WORKERS"
            | "DS4_DOTS3_PREFILL_CHUNK"
            | "DS4_INKLING_NO_LINEAR"
            | "DS4_INKLING_NO_MOE_BATCH"
            | "DS4_INKLING_NO_Q8_BATCH"
            | "DS4_INKLING_NO_MOE_TILE"
            | "DS4_INKLING_NO_LINEAR_TILE"
            | "DS4_INKLING_NO_LOGIT_TILE"
            | "DS4_INKLING_NO_LINEAR_PANEL"
            | "DS4_INKLING_NO_ATTN_GROUP"
            | "DS4_INKLING_NO_Q8_TILE"
            | "DS4_INKLING_NO_SHARED_Q8"
            | "DS4_INKLING_NO_SHARED_TILE"
            | "DS4_INKLING_NO_SHARED_DOWN_TILE"
            | "DS4_INKLING_NO_Q4_TILE"
            | "DS4_INKLING_NO_Q8_ROUTED_TILE"
            | "DS4_INKLING_NO_IQ2_ALIGNED"
            | "DS4_INKLING_NO_IQ2_XS_ALIGNED"
            | "DS4_INKLING_NO_SHARED_SOA"
            | "DS4_INKLING_NO_IQ2_LEAN"
            | "DS4_INKLING_NO_Q3_TILE"
            | "DS4_INKLING_NO_ATTN_TRANSPOSE"
            | "DS4_INKLING_NO_SHARED_COLUMN"
            | "DS4_INKLING_NO_ATTN_PAIR"
            | "DS4_INKLING_NO_Q4_LEAN"
            | "DS4_INKLING_NO_SHARED_PIPE"
            | "DS4_INKLING_NO_IQ2_SLAB"
            | "DS4_INKLING_ATTN_HMMA"
            | "DS4_MIMO2_NO_PREFILL_HMMA"
            | "DS4_MIMO2_NO_PREFILL_ASYNC"
            | "DS4_MIMO2_NO_SWA_HMMA"
            | "DS4_MIMO2_SWA_DECODE"
            | "DS4_MIMO2_SWA_VEC"
            | "DS4_MIMO2_ROUTER_WARP"
            | "DS4_MIMO2_DFLASH_CPU"
            | "DS4_MIMO2_SWIGLU_Q8"
            | "DS4_MIMO2_SUM_RESIDUAL"
            | "DS4_MIMO2_ATTN_RESIDUAL"
            | "DS4_MIMO2_GATEUP_BOUNDED"
            | "DS4_MIMO2_INPUT_Q8_COMPACT"
            | "DS4_MIMO2_DOWN_PIPE64"
            | "DS4_NAIVE_DECODE_SCORES"
            | "DS4_NAIVE_SWA_PREFILL_SCORES"
            | "DS4_NAIVE_DSA_DECODE_TILE"
            | "DS4_NAIVE_DSA_DIRECT"
            | "DS4_NAIVE_SWIGLU_Q8"
            | "DS4_NAIVE_INDEX_PACK"
            | "DS4_NAIVE_INDEX_U2"
            | "DS4_NAIVE_SWA_DECODE_UNIT"
            | "DS4_NAIVE_SWA_RING_WALK"
            | "DS4_NAIVE_ROUTER_WARP"
            | "DS4_NAIVE_SUM_ADD"
            | "DS4_INKLING_PREFILL_CHUNK"
            | "DS4_CUDA_SOLAR_GQA_CHUNK"
            | "DS4_FATTN_HMMA_LDSM"
            | "DS4_SOLAR_FATTN_GQA2"
            | "DS4_SOLAR_FATTN_WS"
            | "DS4_STEP37_PREFILL_CHUNK"
            | "DS4_STEP37_NO_SWA_HMMA"
            | "DS4_EXAONE_PREFILL_GQA"
            | "DS4_LING3VL_PREFILL_CHUNK"
            | "DS4_LING3VL_NO_BF16_VEC"
            | "DS4_LING3VL_NO_MLA_HMMA"
            | "DS4_LING3VL_NO_BF16_REUSE"
            | "DS4_LING3VL_NO_BF16_PAIR"
            | "DS4_LING3VL_NO_GEMV_XREG"
            | "DS4_LING3VL_NO_F32_VEC"
            | "DS4_LING3VL_NO_MOE_FUSE"
            | "DS4_MMQ_Q5_PAIR"
            | "DS4_MMQ_VEC_SANITIZE"
            | "DS4_CUDA_LAYER_GRAPHS"
    )
}

pub fn validate(key: &str, value: &str, family: &str) -> Result<(), String> {
    if key.starts_with("DS4_GEMMA4_") {
        if family.to_ascii_lowercase() != "gemma4_vq" {
            return Err(format!(
                "unsupported experiment control/value for {family}: {key}={value}"
            ));
        }
        return validate_gemma4(key, value)
            .unwrap_or_else(|| Err(format!("unknown Gemma-4 control {key}")));
    }
    let n = value
        .parse::<u32>()
        .map_err(|_| format!("{key}: expected a positive integer"))?;
    let family = family.to_ascii_lowercase();
    let valid = match key {
        "DS4_QWEN_PREFILL_CHUNK" => family.starts_with("qwen") && (1..=16384).contains(&n),
        "DS4_IQUEST_PREFILL_CHUNK" => family == "iquest-q1" && (1..=8192).contains(&n),
        "DS4_QWEN_PLE_WORKERS" => family.starts_with("qwen") && (1..=64).contains(&n),
        "DS4_DOTS3_PREFILL_CHUNK" => family.starts_with("dots") && (1..=8192).contains(&n),
        // Native diagnostic switches test presence; "0" would still disable.
        "DS4_INKLING_NO_LINEAR"
        | "DS4_INKLING_NO_MOE_BATCH"
        | "DS4_INKLING_NO_Q8_BATCH"
        | "DS4_INKLING_NO_MOE_TILE"
        | "DS4_INKLING_NO_LINEAR_TILE"
        | "DS4_INKLING_NO_LOGIT_TILE"
        | "DS4_INKLING_NO_LINEAR_PANEL"
        | "DS4_INKLING_NO_ATTN_GROUP"
        | "DS4_INKLING_NO_Q8_TILE"
        | "DS4_INKLING_NO_SHARED_Q8"
        | "DS4_INKLING_NO_SHARED_TILE"
        | "DS4_INKLING_NO_SHARED_DOWN_TILE"
        | "DS4_INKLING_NO_Q4_TILE"
        | "DS4_INKLING_NO_Q8_ROUTED_TILE"
        | "DS4_INKLING_NO_IQ2_ALIGNED"
        | "DS4_INKLING_NO_IQ2_XS_ALIGNED"
        | "DS4_INKLING_NO_SHARED_SOA"
        | "DS4_INKLING_NO_IQ2_LEAN"
        | "DS4_INKLING_NO_Q3_TILE"
        | "DS4_INKLING_NO_ATTN_TRANSPOSE"
        | "DS4_INKLING_NO_SHARED_COLUMN"
        | "DS4_INKLING_NO_ATTN_PAIR"
        | "DS4_INKLING_NO_Q4_LEAN"
        | "DS4_INKLING_NO_SHARED_PIPE"
        | "DS4_INKLING_NO_IQ2_SLAB"
        | "DS4_INKLING_ATTN_HMMA" => family == "inkling" && value == "1",
        "DS4_MIMO2_NO_PREFILL_HMMA" | "DS4_MIMO2_NO_PREFILL_ASYNC" | "DS4_MIMO2_NO_SWA_HMMA" => {
            family == "mimo2" && value == "1"
        }
        "DS4_MIMO2_SWA_DECODE" | "DS4_MIMO2_SWA_VEC" | "DS4_MIMO2_ROUTER_WARP" => {
            family == "mimo2" && matches!(value, "0" | "1")
        }
        "DS4_MIMO2_DFLASH_CPU" => family == "mimo2" && value == "1",
        "DS4_MIMO2_SWIGLU_Q8" => family == "mimo2" && matches!(value, "0" | "1"),
        "DS4_MIMO2_SUM_RESIDUAL"
        | "DS4_MIMO2_ATTN_RESIDUAL"
        | "DS4_MIMO2_GATEUP_BOUNDED"
        | "DS4_MIMO2_INPUT_Q8_COMPACT"
        | "DS4_MIMO2_DOWN_PIPE64" => family == "mimo2" && matches!(value, "0" | "1"),
        "DS4_NAIVE_DECODE_SCORES"
        | "DS4_NAIVE_SWA_PREFILL_SCORES"
        | "DS4_NAIVE_DSA_DECODE_TILE"
        | "DS4_NAIVE_DSA_DIRECT"
        | "DS4_NAIVE_SWIGLU_Q8"
        | "DS4_NAIVE_INDEX_PACK"
        | "DS4_NAIVE_INDEX_U2"
        | "DS4_NAIVE_SWA_DECODE_UNIT"
        | "DS4_NAIVE_SWA_RING_WALK"
        | "DS4_NAIVE_ROUTER_WARP"
        | "DS4_NAIVE_SUM_ADD" => family == "naive_n05_flash" && matches!(value, "0" | "1"),
        "DS4_INKLING_PREFILL_CHUNK" => family == "inkling" && (1..=8192).contains(&n),
        "DS4_CUDA_SOLAR_GQA_CHUNK" => {
            family.starts_with("solar") && [64, 128, 256, 512, 1024, 2048].contains(&n)
        }
        "DS4_FATTN_HMMA_LDSM" | "DS4_SOLAR_FATTN_GQA2" | "DS4_SOLAR_FATTN_WS" => {
            family == "solar-open2" && matches!(value, "0" | "1")
        }
        "DS4_STEP37_PREFILL_CHUNK" => family.starts_with("step") && (1..=4096).contains(&n),
        "DS4_LING3VL_PREFILL_CHUNK" => {
            (family.starts_with("ling") || family == "bailingmoe3") && (1..=4096).contains(&n)
        }
        "DS4_LING3VL_NO_BF16_VEC"
        | "DS4_LING3VL_NO_MLA_HMMA"
        | "DS4_LING3VL_NO_BF16_REUSE"
        | "DS4_LING3VL_NO_BF16_PAIR"
        | "DS4_LING3VL_NO_GEMV_XREG"
        | "DS4_LING3VL_NO_F32_VEC"
        | "DS4_LING3VL_NO_MOE_FUSE"
        | "DS4_MMQ_VEC_SANITIZE" => {
            (family.starts_with("ling") || family == "bailingmoe3") && value == "1"
        }
        "DS4_MMQ_Q5_PAIR" => {
            (family.starts_with("ling") || family == "bailingmoe3") && value == "0"
        }
        "DS4_CUDA_LAYER_GRAPHS" => {
            (family.starts_with("ling") || family == "bailingmoe3") && matches!(value, "0" | "1")
        }
        "DS4_STEP37_NO_SWA_HMMA" => family.starts_with("step") && value == "1",
        "DS4_EXAONE_PREFILL_GQA" => {
            (family.starts_with("step") || family.starts_with("exaone") || family.starts_with("k2"))
                && value == "0"
        }
        _ => false,
    };
    if !valid {
        return Err(format!(
            "unsupported experiment control/value for {family}: {key}={value}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn gemma4_surface_is_sweepable_with_the_host_bounds() {
        for key in [
            "DS4_GEMMA4_ATTN_SPLIT_SPAN",
            "DS4_GEMMA4_ATTN_SPLIT_MMA",
            "DS4_GEMMA4_CPU_MOE",
            "DS4_GEMMA4_CPU_MOE_THREADS",
            "DS4_GEMMA4_MOE_HOST",
            "DS4_GEMMA4_MOE_MISS",
            "DS4_GEMMA4_PREFILL_CHUNK",
            "DS4_GEMMA4_FULL_KV",
            "DS4_GEMMA4_VQ_GEMV_THREADS",
        ] {
            assert!(super::tunable(key), "{key}");
        }
        assert!(!super::tunable("DS4_GEMMA4_DEBUG"));
        assert!(super::validate("DS4_GEMMA4_ATTN_SPLIT_SPAN", "64", "gemma4_vq").is_ok());
        assert!(super::validate("DS4_GEMMA4_ATTN_SPLIT_SPAN", "8", "gemma4_vq").is_err());
        assert!(super::validate("DS4_GEMMA4_ATTN_SPLIT_SPAN", "64", "glm5_next").is_err());
        assert!(super::validate("DS4_GEMMA4_MOE_MISS", "cpu", "gemma4_vq").is_ok());
        assert!(super::validate("DS4_GEMMA4_MOE_MISS", "cpux", "gemma4_vq").is_err());
        assert!(super::validate("DS4_GEMMA4_FULL_KV", "fp8", "gemma4_vq").is_ok());
        assert!(super::validate("DS4_GEMMA4_FULL_KV", "fp8x", "gemma4_vq").is_err());
        assert!(super::validate("DS4_GEMMA4_VQ_GEMV_THREADS", "384", "gemma4_vq").is_ok());
        assert!(super::validate("DS4_GEMMA4_VQ_GEMV_THREADS", "33", "gemma4_vq").is_err());
        assert!(super::validate("DS4_GEMMA4_CPU_MOE", "on", "gemma4_vq").is_ok());
        assert!(super::validate("DS4_GEMMA4_CPU_MOE", "yes", "gemma4_vq").is_err());
    }

    #[test]
    fn iquest_chunk_is_scoped_and_bounded() {
        assert!(super::tunable("DS4_IQUEST_PREFILL_CHUNK"));
        assert!(super::validate("DS4_IQUEST_PREFILL_CHUNK", "8192", "IQuest-Q1").is_ok());
        assert!(super::validate("DS4_IQUEST_PREFILL_CHUNK", "0", "iquest-q1").is_err());
        assert!(super::validate("DS4_IQUEST_PREFILL_CHUNK", "8193", "iquest-q1").is_err());
        assert!(super::validate("DS4_IQUEST_PREFILL_CHUNK", "128", "qwen4exp").is_err());
    }
    use super::*;

    #[test]
    fn naive_attention_controls() {
        for key in [
            "DS4_NAIVE_DECODE_SCORES",
            "DS4_NAIVE_SWA_PREFILL_SCORES",
            "DS4_NAIVE_DSA_DECODE_TILE",
            "DS4_NAIVE_DSA_DIRECT",
            "DS4_NAIVE_SWIGLU_Q8",
            "DS4_NAIVE_INDEX_PACK",
            "DS4_NAIVE_INDEX_U2",
            "DS4_NAIVE_SWA_DECODE_UNIT",
            "DS4_NAIVE_SWA_RING_WALK",
            "DS4_NAIVE_ROUTER_WARP",
            "DS4_NAIVE_SUM_ADD",
        ] {
            assert!(tunable(key));
            for value in ["0", "1"] {
                assert!(validate(key, value, "naive_n05_flash").is_ok());
            }
            for value in ["2", "01", "true", ""] {
                assert!(validate(key, value, "naive_n05_flash").is_err());
            }
            assert!(validate(key, "1", "mimo2").is_err());
        }
    }

    #[test]
    fn solar_attention_controls() {
        for key in [
            "DS4_FATTN_HMMA_LDSM",
            "DS4_SOLAR_FATTN_GQA2",
            "DS4_SOLAR_FATTN_WS",
        ] {
            assert!(tunable(key));
            for value in ["0", "1"] {
                assert!(validate(key, value, "solar-open2").is_ok());
            }
            for value in ["2", "01", "true", ""] {
                assert!(validate(key, value, "solar-open2").is_err());
            }
            assert!(validate(key, "1", "inkling").is_err());
        }
        assert!(!tunable("DS4_METAL_PREFILL_CHUNK"));
        assert!(!tunable("DS4_SOLAR_KV_FORMAT"));
    }

    #[test]
    fn inkling_controls() {
        for key in [
            "DS4_INKLING_NO_LINEAR",
            "DS4_INKLING_NO_MOE_BATCH",
            "DS4_INKLING_NO_Q8_BATCH",
            "DS4_INKLING_NO_MOE_TILE",
            "DS4_INKLING_NO_LINEAR_TILE",
            "DS4_INKLING_NO_LOGIT_TILE",
            "DS4_INKLING_NO_LINEAR_PANEL",
            "DS4_INKLING_NO_ATTN_GROUP",
            "DS4_INKLING_NO_Q8_TILE",
            "DS4_INKLING_NO_SHARED_Q8",
            "DS4_INKLING_NO_SHARED_TILE",
            "DS4_INKLING_NO_SHARED_DOWN_TILE",
            "DS4_INKLING_NO_Q4_TILE",
            "DS4_INKLING_NO_Q8_ROUTED_TILE",
            "DS4_INKLING_NO_IQ2_ALIGNED",
            "DS4_INKLING_NO_IQ2_XS_ALIGNED",
            "DS4_INKLING_NO_SHARED_SOA",
            "DS4_INKLING_NO_IQ2_LEAN",
            "DS4_INKLING_NO_Q3_TILE",
            "DS4_INKLING_NO_ATTN_TRANSPOSE",
            "DS4_INKLING_NO_SHARED_COLUMN",
            "DS4_INKLING_NO_ATTN_PAIR",
            "DS4_INKLING_NO_Q4_LEAN",
            "DS4_INKLING_NO_SHARED_PIPE",
            "DS4_INKLING_NO_IQ2_SLAB",
            "DS4_INKLING_ATTN_HMMA",
        ] {
            assert!(tunable(key));
            assert!(validate(key, "1", "Inkling").is_ok());
            for value in ["0", "2", "01", "true", ""] {
                assert!(validate(key, value, "inkling").is_err());
            }
            for family in ["qwen", "inkling-other", ""] {
                assert!(validate(key, "1", family).is_err());
            }
        }
        assert!(tunable("DS4_INKLING_PREFILL_CHUNK"));
        for value in ["1", "512", "1024", "2048", "2049", "8192"] {
            assert!(validate("DS4_INKLING_PREFILL_CHUNK", value, "inkling").is_ok());
        }
        for value in ["0", "8193", "x"] {
            assert!(validate("DS4_INKLING_PREFILL_CHUNK", value, "inkling").is_err());
        }
        assert!(validate("DS4_INKLING_PREFILL_CHUNK", "512", "qwen").is_err());
        assert!(!tunable("DS4_INKLING_UNKNOWN"));
    }

    #[test]
    fn mimo2_prefill_hmma_kill() {
        assert!(tunable("DS4_MIMO2_NO_PREFILL_HMMA"));
        assert!(validate("DS4_MIMO2_NO_PREFILL_HMMA", "1", "mimo2").is_ok());
        assert!(validate("DS4_MIMO2_NO_PREFILL_HMMA", "1", "MiMo2").is_ok());
        for value in ["0", "2", "01", "true", ""] {
            assert!(validate("DS4_MIMO2_NO_PREFILL_HMMA", value, "mimo2").is_err());
        }
        assert!(validate("DS4_MIMO2_NO_PREFILL_HMMA", "1", "inkling").is_err());
        assert!(tunable("DS4_MIMO2_NO_PREFILL_ASYNC"));
        assert!(validate("DS4_MIMO2_NO_PREFILL_ASYNC", "1", "mimo2").is_ok());
        assert!(validate("DS4_MIMO2_NO_PREFILL_ASYNC", "0", "mimo2").is_err());
        assert!(tunable("DS4_MIMO2_NO_SWA_HMMA"));
        assert!(validate("DS4_MIMO2_NO_SWA_HMMA", "1", "mimo2").is_ok());
        assert!(validate("DS4_MIMO2_NO_SWA_HMMA", "0", "mimo2").is_err());
    }

    #[test]
    fn mimo2_decode_controls() {
        for key in [
            "DS4_MIMO2_SWA_DECODE",
            "DS4_MIMO2_SWA_VEC",
            "DS4_MIMO2_ROUTER_WARP",
        ] {
            assert!(tunable(key));
            for value in ["0", "1"] {
                assert!(validate(key, value, "mimo2").is_ok());
            }
            for value in ["2", "01", "true", ""] {
                assert!(validate(key, value, "mimo2").is_err());
            }
            assert!(validate(key, "1", "qwen").is_err());
        }
    }

    #[test]
    fn mimo2_swiglu_control() {
        let key = "DS4_MIMO2_SWIGLU_Q8";
        assert!(tunable(key));
        assert!(validate(key, "0", "mimo2").is_ok());
        assert!(validate(key, "1", "mimo2").is_ok());
        assert!(validate(key, "2", "mimo2").is_err());
        assert!(validate(key, "1", "qwen").is_err());
    }

    #[test]
    fn mimo2_fusion_controls() {
        for key in [
            "DS4_MIMO2_SUM_RESIDUAL",
            "DS4_MIMO2_ATTN_RESIDUAL",
            "DS4_MIMO2_GATEUP_BOUNDED",
            "DS4_MIMO2_INPUT_Q8_COMPACT",
            "DS4_MIMO2_DOWN_PIPE64",
        ] {
            assert!(tunable(key), "{key} is missing from matched comparisons");
            for value in ["0", "1"] {
                assert!(validate(key, value, "mimo2").is_ok());
                assert!(validate(key, value, "MiMo2").is_ok());
            }
            for value in ["2", "01", "true", ""] {
                assert!(validate(key, value, "mimo2").is_err());
            }
            assert!(validate(key, "1", "qwen").is_err());
        }
    }

    #[test]
    fn mimo2_dflash_control() {
        let key = "DS4_MIMO2_DFLASH_CPU";
        assert!(tunable(key));
        assert!(validate(key, "1", "mimo2").is_ok());
        assert!(validate(key, "0", "mimo2").is_err());
        assert!(validate(key, "1", "qwen").is_err());
    }

    #[test]
    fn step37_controls() {
        assert!(tunable("DS4_STEP37_PREFILL_CHUNK"));
        assert!(tunable("DS4_STEP37_NO_SWA_HMMA"));
        assert!(tunable("DS4_EXAONE_PREFILL_GQA"));
        assert!(validate("DS4_EXAONE_PREFILL_GQA", "0", "step37").is_ok());
        assert!(validate("DS4_EXAONE_PREFILL_GQA", "0", "k2-horizon").is_ok());
        assert!(validate("DS4_EXAONE_PREFILL_GQA", "0", "k2").is_ok());
        assert!(validate("DS4_EXAONE_PREFILL_GQA", "1", "step37").is_err());
        assert!(validate("DS4_EXAONE_PREFILL_GQA", "0", "qwen").is_err());
        for value in ["1", "512", "1024", "2048", "4096"] {
            assert!(validate("DS4_STEP37_PREFILL_CHUNK", value, "step37").is_ok());
        }
        for value in ["0", "4097", "x"] {
            assert!(validate("DS4_STEP37_PREFILL_CHUNK", value, "step37").is_err());
        }
        assert!(validate("DS4_STEP37_PREFILL_CHUNK", "512", "qwen").is_err());
        assert!(validate("DS4_STEP37_NO_SWA_HMMA", "1", "step37").is_ok());
        assert!(validate("DS4_STEP37_NO_SWA_HMMA", "1", "qwen").is_err());
        for value in ["0", "2", ""] {
            assert!(validate("DS4_STEP37_NO_SWA_HMMA", value, "step37").is_err());
        }
    }

    #[test]
    fn ling3vl_controls() {
        assert!(tunable("DS4_LING3VL_PREFILL_CHUNK"));
        for value in ["1", "512", "1024", "2048", "4096"] {
            assert!(validate("DS4_LING3VL_PREFILL_CHUNK", value, "ling3vl").is_ok());
            assert!(validate("DS4_LING3VL_PREFILL_CHUNK", value, "ling").is_ok());
        }
        for value in ["0", "4097", "x"] {
            assert!(validate("DS4_LING3VL_PREFILL_CHUNK", value, "ling3vl").is_err());
        }
        assert!(validate("DS4_LING3VL_PREFILL_CHUNK", "2048", "qwen").is_err());
        for key in [
            "DS4_LING3VL_NO_BF16_VEC",
            "DS4_LING3VL_NO_MLA_HMMA",
            "DS4_LING3VL_NO_BF16_REUSE",
            "DS4_LING3VL_NO_BF16_PAIR",
            "DS4_LING3VL_NO_GEMV_XREG",
            "DS4_LING3VL_NO_F32_VEC",
            "DS4_LING3VL_NO_MOE_FUSE",
            "DS4_MMQ_VEC_SANITIZE",
        ] {
            assert!(tunable(key));
            assert!(validate(key, "1", "ling3vl").is_ok());
            assert!(validate(key, "0", "ling3vl").is_err());
            assert!(validate(key, "1", "qwen").is_err());
        }
        assert!(tunable("DS4_MMQ_Q5_PAIR"));
        assert!(validate("DS4_MMQ_Q5_PAIR", "0", "ling3vl").is_ok());
        assert!(validate("DS4_MMQ_Q5_PAIR", "1", "ling3vl").is_err());
        assert!(tunable("DS4_CUDA_LAYER_GRAPHS"));
        assert!(validate("DS4_CUDA_LAYER_GRAPHS", "0", "ling3vl").is_ok());
        assert!(validate("DS4_CUDA_LAYER_GRAPHS", "1", "ling3vl").is_ok());
        assert!(validate("DS4_CUDA_LAYER_GRAPHS", "0", "qwen").is_err());
    }
}
