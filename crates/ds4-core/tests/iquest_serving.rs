//! Model-free IQuest serving and native-allocation contracts.
use ds4_core::{
    caps_from_shape, fill_quote_facts, resolve_plan, shape_for_variant, Backend, ChunkFence,
    Distribution, EngineFacts, IssueLevel, LaneMode, MaxSeqs, MtpMode, PrefixReuse, QuoteHost,
    ResolvedPlan, ReuseKind, ServingRequest, Support, Variant,
};

const CONTEXT: i32 = 524_288;
const WEIGHTS: u64 = 87_975_869_440;
const GIB: u64 = 1 << 30;

fn request() -> ServingRequest {
    ServingRequest {
        backend: Backend::Cuda,
        ctx: 8192,
        max_seqs: MaxSeqs::Fixed(2),
        prefix_reuse: PrefixReuse::Off,
        mtp_mode: MtpMode::Off,
        native_chunk: Some(128),
        ..ServingRequest::default()
    }
}

fn facts(req: &ServingRequest) -> EngineFacts {
    let shape = shape_for_variant(Variant::IQuestQ1);
    let mut facts = EngineFacts::default();
    fill_quote_facts(
        &mut facts,
        req,
        caps_from_shape(shape),
        Some(shape),
        QuoteHost {
            weights_bytes: WEIGHTS,
            mtp_bytes: 0,
            available_bytes: 128 * GIB,
            native_chunk: None,
            vision: false,
        },
    );
    facts
}

fn plan(req: &ServingRequest, facts: &EngineFacts) -> ResolvedPlan {
    let caps = caps_from_shape(shape_for_variant(Variant::IQuestQ1));
    resolve_plan(req, Some(caps), facts)
}

fn no_errors(plan: &ResolvedPlan) {
    assert!(
        plan.issues
            .iter()
            .all(|issue| issue.level != IssueLevel::Error),
        "{:?}",
        plan.issues
    );
}

fn has_error(plan: &ResolvedPlan, code: &str) {
    assert!(
        plan.issues
            .iter()
            .any(|issue| issue.level == IssueLevel::Error && issue.code == code),
        "{code}: {:?}",
        plan.issues
    );
}

#[test]
fn recursive_drafts_both_lanes() {
    for lane in [LaneMode::Auto, LaneMode::Serial] {
        for draft in 2..=7 {
            let req = ServingRequest {
                lane,
                max_seqs: MaxSeqs::Fixed(1),
                mtp_mode: MtpMode::On,
                mtp_draft: Some(draft),
                ..request()
            };
            let resolved = plan(&req, &facts(&req));
            no_errors(&resolved);
            assert_eq!(resolved.effective.mtp_mode, MtpMode::On);
            assert_eq!(resolved.effective.mtp_draft, Some(draft));
            assert!(resolved.effective.mtp_weights);
            assert_eq!(resolved.qualified.mtp, Support::Present);
        }
    }
    for draft in [0, 1, 8] {
        let req = ServingRequest {
            mtp_mode: MtpMode::On,
            mtp_draft: Some(draft),
            ..request()
        };
        let resolved = plan(&req, &facts(&req));
        has_error(&resolved, "mtp_draft");
        assert_eq!(resolved.effective.mtp_mode, MtpMode::Off);
    }
}

#[test]
fn refuses_sidecars_and_hosts() {
    let req = ServingRequest {
        mtp_mode: MtpMode::On,
        mtp_path: Some("draft.gguf".into()),
        ..request()
    };
    has_error(&plan(&req, &facts(&req)), "mtp_contract");
    for backend in [Backend::Cpu, Backend::Metal] {
        let req = ServingRequest {
            backend,
            ..request()
        };
        has_error(&plan(&req, &facts(&req)), "family_host");
    }
    let req = ServingRequest {
        distribution: Distribution::Sliced,
        ..request()
    };
    has_error(&plan(&req, &facts(&req)), "family_host");
}

#[test]
fn context_and_bank_admission() {
    let mut req = ServingRequest {
        ctx: CONTEXT,
        max_seqs: MaxSeqs::Fixed(1),
        ..request()
    };
    let resolved = plan(&req, &facts(&req));
    no_errors(&resolved);
    assert_eq!(resolved.effective.ctx, CONTEXT);
    assert_eq!(resolved.qualified.ctx, None);
    req.ctx += 1;
    has_error(&plan(&req, &facts(&req)), "ctx_unavailable");
    req.ctx = 0;
    has_error(&plan(&req, &facts(&req)), "ctx_invalid");

    let req = request();
    let mut fitted = facts(&req);
    fitted.banks_fitted = Some(1);
    has_error(&plan(&req, &fitted), "banks_not_fitted");
    fitted.banks_fitted = Some(2);
    let resolved = plan(&req, &fitted);
    no_errors(&resolved);
    assert_eq!(resolved.effective.max_seqs, 2);
    assert_eq!(resolved.qualified.banks, Support::Present);
}

#[test]
fn cache_needs_runtime() {
    let req = ServingRequest {
        prefix_reuse: PrefixReuse::Partial,
        kv_disk_dir: Some("cache".into()),
        kv_disk_space_mb: Some(8192),
        ..request()
    };
    let mut state = facts(&req);
    let resolved = plan(&req, &state);
    no_errors(&resolved);
    assert_eq!(resolved.effective.prefix_reuse, ReuseKind::Partial);
    assert!(resolved.effective.disk);
    assert_eq!(resolved.qualified.disk, Support::Present);
    state.partial_reuse = Some(false);
    has_error(&plan(&req, &state), "partial_runtime");
    state.disk_ready = Some(false);
    has_error(&plan(&req, &state), "disk_open");
}

#[test]
fn native_chunks_are_bounded() {
    for chunk in [1, 4, 128, 8192] {
        let req = ServingRequest {
            native_chunk: Some(chunk),
            chunk_fence: ChunkFence::Off,
            ..request()
        };
        let state = facts(&req);
        assert_eq!(state.native_chunk, Some(chunk));
        no_errors(&plan(&req, &state));
    }
    for chunk in [0, 8193] {
        let req = ServingRequest {
            native_chunk: Some(chunk),
            ..request()
        };
        has_error(&plan(&req, &facts(&req)), "native_chunk");
    }
}

#[test]
fn quote_matches_native_512k() {
    let req = ServingRequest {
        ctx: CONTEXT,
        max_seqs: MaxSeqs::Fixed(1),
        native_chunk: Some(4),
        ..request()
    };
    let state = facts(&req);
    // B300's recorded 524288-capacity, cap=4 bank allocated these tensors.
    assert_eq!(
        state.per_bank_bytes.unwrap() + state.scratch_bytes.unwrap(),
        29_090_663_072
    );
    assert_eq!(state.checkpoint_pool_bytes, Some(0));
    assert_eq!(state.shared_weights_bytes, Some(WEIGHTS));
    let target_kv = (25 * CONTEXT as u64 + 63 * (4096 + 3)) * 2176;
    assert_eq!(
        state.per_bank_bytes,
        Some(target_kv + 519 * 2176 + 3072 * 4)
    );

    let req = ServingRequest {
        max_seqs: MaxSeqs::Fixed(2),
        ..req
    };
    has_error(&plan(&req, &facts(&req)), "banks_not_quoted");
}

#[test]
fn quote_counts_checkpoint_pool() {
    let req = ServingRequest {
        prefix_reuse: PrefixReuse::Partial,
        ..request()
    };
    let state = facts(&req);
    assert_eq!(
        state.checkpoint_pool_bytes,
        Some(8 * ((63 * 4096 + 512) * 2176 + 3072 * 4))
    );
    let single = ServingRequest {
        max_seqs: MaxSeqs::Fixed(1),
        ..req.clone()
    };
    assert_eq!(state.scratch_bytes, facts(&single).scratch_bytes);
    assert_eq!(
        state.checkpoint_pool_bytes,
        facts(&single).checkpoint_pool_bytes
    );
}
