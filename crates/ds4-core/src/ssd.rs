//! Common SSD expert-cache options; native owns storage and execution.

use crate::{Backend, DistributedConfig, Error, ModelFamily, ModelOpenOption, OpenTuning, Result};

const GIB: u64 = 1 << 30;
const CACHE_ARG_ERROR: &str =
    "--ssd-streaming-cache-experts must be a positive count or <number>GB";
const COPY_ENV_KEYS: [&str; 4] = [
    "DS4_MODEL_ANON_HUGE",
    "DS4_CUDA_WEIGHT_IPC_MANIFEST",
    "DS4_CUDA_COPY_MODEL",
    "DS4_CUDA_COPY_MODEL_CHUNKED",
];

impl ModelOpenOption {
    /// A bare count is global expert capacity; GB denotes GiB, as upstream.
    pub fn ssd_cache(value: &str) -> Result<Self> {
        let invalid = || Error {
            code: 1,
            message: CACHE_ARG_ERROR.into(),
        };
        if let Some(number) = value
            .strip_suffix("GB")
            .or_else(|| value.strip_suffix("gb"))
        {
            let gib = number.parse::<f64>().map_err(|_| invalid())?;
            let bytes = gib * GIB as f64;
            if !bytes.is_finite() || bytes < 1.0 || bytes >= u64::MAX as f64 {
                return Err(invalid());
            }
            return Ok(Self::SsdCacheBytes(bytes as u64));
        }
        let count = value.parse::<u32>().map_err(|_| invalid())?;
        if count == 0 {
            return Err(invalid());
        }
        Ok(Self::SsdCacheExperts(count))
    }
}

/// Run the same SSD admission on metadata-only server preflight and open.
pub fn check_ssd_options(
    options: &[ModelOpenOption],
    family: Option<ModelFamily>,
    backend: Backend,
    distributed: Option<&DistributedConfig>,
) -> Result<()> {
    check_tuning(&crate::open_tuning(options)?, family, backend, distributed)
}

pub(super) fn check_tuning(
    tuning: &OpenTuning,
    family: Option<ModelFamily>,
    backend: Backend,
    distributed: Option<&DistributedConfig>,
) -> Result<()> {
    if !tuning.ssd_streaming {
        return Ok(());
    }
    if family != Some(ModelFamily::Glm53) || backend != Backend::Cuda || distributed.is_some() {
        return Err(Error {
            code: 1,
            message: "--ssd-streaming requires one full GLM-5.3 CUDA model".into(),
        });
    }
    // These native modes materialize/import the full model before streaming.
    if tuning.warm_weights
        || COPY_ENV_KEYS
            .iter()
            .any(|key| std::env::var_os(key).is_some())
    {
        return Err(Error {
            code: 1,
            message:
                "SSD streaming conflicts with eager warming, anonymous model copies and weight IPC"
                    .into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_counts_and_bytes() {
        assert_eq!(
            ModelOpenOption::ssd_cache("8").unwrap(),
            ModelOpenOption::SsdCacheExperts(8)
        );
        assert_eq!(
            ModelOpenOption::ssd_cache("1.5GB").unwrap(),
            ModelOpenOption::SsdCacheBytes(GIB + GIB / 2)
        );
        for invalid in [
            "",
            "0",
            "-1",
            "4294967296",
            "0GB",
            "NaNGB",
            "infGB",
            "1e30GB",
            "1e-20GB",
            "16MB",
        ] {
            assert!(ModelOpenOption::ssd_cache(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn cache_requires_streaming() {
        for option in [
            ModelOpenOption::SsdStreamingCold,
            ModelOpenOption::SsdCacheExperts(8),
            ModelOpenOption::SsdCacheBytes(GIB),
        ] {
            assert!(
                check_ssd_options(&[option], Some(ModelFamily::Glm53), Backend::Cuda, None)
                    .is_err()
            );
        }
    }

    #[test]
    fn cache_budget_is_exclusive() {
        let options = [
            ModelOpenOption::SsdStreaming,
            ModelOpenOption::SsdCacheExperts(8),
            ModelOpenOption::SsdCacheBytes(GIB),
        ];
        assert!(
            check_ssd_options(&options, Some(ModelFamily::Glm53), Backend::Cuda, None).is_err()
        );
    }

    #[test]
    fn ssd_warming_conflicts() {
        let options = [ModelOpenOption::SsdStreaming, ModelOpenOption::WarmWeights];
        assert!(
            check_ssd_options(&options, Some(ModelFamily::Glm53), Backend::Cuda, None).is_err()
        );
    }

    #[test]
    fn admission_is_glm_cuda_only() {
        let options = [ModelOpenOption::SsdStreaming];
        assert!(check_ssd_options(&options, Some(ModelFamily::Glm53), Backend::Cuda, None).is_ok());
        for backend in [Backend::Cpu, Backend::Metal] {
            assert!(check_ssd_options(&options, Some(ModelFamily::Glm53), backend, None).is_err());
        }
        for family in [
            None,
            Some(ModelFamily::DeepSeek4),
            Some(ModelFamily::Qwen4Exp),
        ] {
            assert!(check_ssd_options(&options, family, Backend::Cuda, None).is_err());
        }
        assert!(check_ssd_options(&[], None, Backend::Cpu, None).is_ok());
    }
}
