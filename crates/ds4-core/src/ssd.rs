//! Common SSD expert-cache options; native owns storage and execution.

use crate::{Backend, DistributedConfig, Error, ModelFamily, ModelOpenOption, OpenTuning, Result};

const GIB: u64 = 1 << 30;
const CACHE_ARG_ERROR: &str =
    "--ssd-streaming-cache-experts must be auto, a positive count or <number>GB";
const AUTO_BUDGET_ERROR: &str =
    "SSD Auto cache requires ServingBudget with the requested context and banks";
const COPY_ENV_KEYS: [&str; 4] = [
    "DS4_MODEL_ANON_HUGE",
    "DS4_CUDA_WEIGHT_IPC_MANIFEST",
    "DS4_CUDA_COPY_MODEL",
    "DS4_CUDA_COPY_MODEL_CHUNKED",
];

impl ModelOpenOption {
    /// A bare count is global expert capacity; GB denotes GiB, as upstream.
    pub fn ssd_cache(value: &str) -> Result<Self> {
        if value == "auto" {
            return Ok(Self::SsdCacheAuto);
        }
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
    check_budget(tuning)
}

fn check_budget(tuning: &OpenTuning) -> Result<()> {
    // Auto commits expert memory at open; a later session can shrink rows,
    // but cannot reclaim that cache for its actual context or bank count.
    if tuning.serving_budget.is_some()
        || tuning.ssd_streaming_cache_experts != 0
        || tuning.ssd_streaming_cache_bytes != 0
    {
        return Ok(());
    }
    Err(Error {
        code: 1,
        message: AUTO_BUDGET_ERROR.into(),
    })
}

pub(super) fn resolve_budget(
    tuning: &mut OpenTuning,
    id: &crate::Identified,
    inventory: &crate::TensorInventory,
    backend: Backend,
) -> Result<()> {
    if !tuning.ssd_streaming {
        return Ok(());
    }
    check_budget(tuning)?;
    let Some(mut req) = tuning.serving_budget.clone() else {
        // A forced cache needs no invented workload. Native validates its
        // capacity at open; session/bank creation admits the actual request.
        return Ok(());
    };
    req.backend = backend;
    req.ssd_streaming = true;
    req.ssd_streaming_cold = tuning.ssd_streaming_cold;
    req.ssd_streaming_cache_experts =
        (tuning.ssd_streaming_cache_experts != 0).then_some(tuning.ssd_streaming_cache_experts);
    req.ssd_streaming_cache_bytes =
        (tuning.ssd_streaming_cache_bytes != 0).then_some(tuning.ssd_streaming_cache_bytes);
    req.mtp_draft = Some(tuning.mtp_draft_tokens);
    req.mtp_mode = crate::glm_mtp::mode(
        tuning.mtp_draft_tokens,
        std::env::var("DS4_GLM53_MTP").ok().as_deref(),
        std::env::var("DS4_MTP_SPEC_DISABLE").ok().as_deref(),
    );
    let caps = crate::caps_from_ident(id);
    let mut facts = crate::EngineFacts::default();
    crate::probe_ssd_quote(&mut facts, &req, id.shape, inventory)?;
    crate::attach_host_quote(
        &mut facts,
        &req,
        caps,
        Some(id.shape),
        None,
        None,
        tuning.vision_path.as_deref().map(std::path::Path::new),
        None,
        1,
        None,
        tuning.vision_path.is_some(),
        false,
    );
    let plan = crate::resolve_plan(&req, Some(caps), &facts);
    if plan.has_errors() || plan.quote.is_none() {
        return Err(Error {
            code: 1,
            message: format!("SSD budget rejected: {}", plan.report()),
        });
    }
    tuning.ssd_streaming_cache_experts =
        plan.effective
            .ssd_streaming_cache_experts
            .ok_or_else(|| Error {
                code: 1,
                message: "SSD cache budget unavailable".into(),
            })?;
    tuning.ssd_streaming_cache_bytes = 0;
    if let Some(rows) = plan.effective.native_chunk {
        std::env::set_var("DS4_GLM53_PREFILL_ROWS", rows.to_string());
    }
    crate::ssd_quote::apply_window(plan.effective.prefill_window);
    eprintln!("SSD admission: requested={} rows={:?} effective={} experts {} bytes rows={:?} ctx={} banks={} qualified=unverified",
        if req.ssd_streaming_cache_experts.is_some() || req.ssd_streaming_cache_bytes.is_some() { "fixed" } else { "auto" },
        req.native_chunk, tuning.ssd_streaming_cache_experts,
        plan.effective.ssd_streaming_cache_bytes.unwrap_or(0), plan.effective.native_chunk,
        req.ctx, plan.effective.max_seqs);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_counts_and_bytes() {
        assert_eq!(
            ModelOpenOption::ssd_cache("auto").unwrap(),
            ModelOpenOption::SsdCacheAuto
        );
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
            ModelOpenOption::SsdCacheAuto,
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
        assert!(check_ssd_options(
            &[
                ModelOpenOption::SsdStreaming,
                ModelOpenOption::SsdCacheAuto,
                ModelOpenOption::SsdCacheExperts(8)
            ],
            Some(ModelFamily::Glm53),
            Backend::Cuda,
            None
        )
        .is_err());
    }

    #[test]
    fn ssd_warming_conflicts() {
        let options = [ModelOpenOption::SsdStreaming, ModelOpenOption::WarmWeights];
        assert!(
            check_ssd_options(&options, Some(ModelFamily::Glm53), Backend::Cuda, None).is_err()
        );
    }

    #[test]
    fn auto_requires_serving_budget() {
        let req = crate::ServingRequest {
            ctx: 262144,
            max_seqs: crate::MaxSeqs::Fixed(2),
            mtp_mode: crate::MtpMode::On,
            mtp_draft: Some(3),
            prefix_reuse: crate::PrefixReuse::Partial,
            ..crate::ServingRequest::default()
        };
        for explicit in [false, true] {
            let mut options = vec![ModelOpenOption::SsdStreaming];
            if explicit {
                options.push(ModelOpenOption::SsdCacheAuto);
            }
            let error = check_ssd_options(&options, Some(ModelFamily::Glm53), Backend::Cuda, None)
                .unwrap_err();
            assert!(error.message.contains("ServingBudget"), "{error}");

            options.push(ModelOpenOption::ServingBudget(req.clone()));
            assert!(
                check_ssd_options(&options, Some(ModelFamily::Glm53), Backend::Cuda, None,).is_ok()
            );
        }
        for fixed in [
            ModelOpenOption::SsdCacheExperts(8),
            ModelOpenOption::SsdCacheBytes(GIB),
        ] {
            assert!(check_ssd_options(
                &[ModelOpenOption::SsdStreaming, fixed],
                Some(ModelFamily::Glm53),
                Backend::Cuda,
                None,
            )
            .is_ok());
        }
    }

    #[test]
    fn auto_fails_before_tensor_probe() {
        let id = crate::Identified {
            shape: crate::SHAPE_GLM53_FLASH,
            architecture: None,
            split_count: 1,
            n_kv: 0,
            n_tensors: 0,
            alignment: 32,
            version: 3,
        };
        // Empty metadata must not hide the missing workload or reach sizing.
        let inventory = crate::TensorInventory {
            shards: Vec::new(),
            tensors: Vec::new(),
            data_pos: 0,
            alignment: 32,
            page: 4096,
        };
        let mut tuning = crate::open_tuning(&[ModelOpenOption::SsdStreaming]).unwrap();
        let error = resolve_budget(&mut tuning, &id, &inventory, Backend::Cuda).unwrap_err();
        assert!(error.message.contains("ServingBudget"), "{error}");
        assert_eq!(tuning.ssd_streaming_cache_experts, 0);
        assert_eq!(tuning.ssd_streaming_cache_bytes, 0);
    }

    #[test]
    fn fixed_cache_defers_workload() {
        let id = crate::Identified {
            shape: crate::SHAPE_GLM53_FLASH,
            architecture: None,
            split_count: 1,
            n_kv: 0,
            n_tensors: 0,
            alignment: 32,
            version: 3,
        };
        let inventory = crate::TensorInventory {
            shards: Vec::new(),
            tensors: Vec::new(),
            data_pos: 0,
            alignment: 32,
            page: 4096,
        };
        for fixed in [
            ModelOpenOption::SsdCacheExperts(8),
            ModelOpenOption::SsdCacheBytes(GIB),
        ] {
            let options = [ModelOpenOption::SsdStreaming, fixed];
            let mut tuning = crate::open_tuning(&options).unwrap();
            let count = tuning.ssd_streaming_cache_experts;
            let bytes = tuning.ssd_streaming_cache_bytes;
            // Without a workload, sizing must leave the forced native budget
            // intact. Empty metadata catches any invented serving request.
            resolve_budget(&mut tuning, &id, &inventory, Backend::Cuda).unwrap();
            assert_eq!(tuning.ssd_streaming_cache_experts, count);
            assert_eq!(tuning.ssd_streaming_cache_bytes, bytes);

            tuning.serving_budget = Some(crate::ServingRequest::default());
            assert!(resolve_budget(&mut tuning, &id, &inventory, Backend::Cuda).is_err());
        }
    }

    #[test]
    fn admission_is_glm_cuda_only() {
        let options = [
            ModelOpenOption::SsdStreaming,
            ModelOpenOption::SsdCacheExperts(8),
        ];
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
