// Only controls whose values are consumed by the native execution path may
// vary inside a matched experiment. Memory/residency identity stays fixed.
pub fn tunable(key: &str) -> bool {
    matches!(
        key,
        "DS4_QWEN_PREFILL_CHUNK"
            | "DS4_QWEN_PLE_WORKERS"
            | "DS4_DOTS3_PREFILL_CHUNK"
            | "DS4_INKLING_NO_LINEAR"
            | "DS4_INKLING_NO_MOE_BATCH"
            | "DS4_CUDA_SOLAR_GQA_CHUNK"
    )
}

pub fn validate(key: &str, value: &str, family: &str) -> Result<(), String> {
    let n = value
        .parse::<u32>()
        .map_err(|_| format!("{key}: expected a positive integer"))?;
    let family = family.to_ascii_lowercase();
    let valid = match key {
        "DS4_QWEN_PREFILL_CHUNK" => family.starts_with("qwen") && (1..=16384).contains(&n),
        "DS4_QWEN_PLE_WORKERS" => family.starts_with("qwen") && (1..=64).contains(&n),
        "DS4_DOTS3_PREFILL_CHUNK" => family.starts_with("dots") && (1..=8192).contains(&n),
        // Native diagnostic switches test presence; "0" would still disable.
        "DS4_INKLING_NO_LINEAR" | "DS4_INKLING_NO_MOE_BATCH" => {
            family == "inkling" && value == "1"
        }
        "DS4_CUDA_SOLAR_GQA_CHUNK" => {
            family.starts_with("solar") && [64, 128, 256, 512, 1024, 2048].contains(&n)
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
    use super::*;

    #[test]
    fn inkling_linear_control() {
        for key in ["DS4_INKLING_NO_LINEAR", "DS4_INKLING_NO_MOE_BATCH"] {
            assert!(tunable(key));
            assert!(validate(key, "1", "Inkling").is_ok());
            for value in ["0", "2", "01", "true", ""] {
                assert!(validate(key, value, "inkling").is_err());
            }
            for family in ["qwen", "inkling-other", ""] {
                assert!(validate(key, "1", family).is_err());
            }
        }
        assert!(!tunable("DS4_INKLING_UNKNOWN"));
    }
}
