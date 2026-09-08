// Only controls whose values are consumed by the native execution path may
// vary inside a matched experiment. Memory/residency identity stays fixed.
pub fn tunable(key: &str) -> bool {
    matches!(
        key,
        "DS4_QWEN_PREFILL_CHUNK"
            | "DS4_QWEN_PLE_WORKERS"
            | "DS4_DOTS3_PREFILL_CHUNK"
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
