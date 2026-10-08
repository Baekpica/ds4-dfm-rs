//! Mirror the native GLM cache padding and validated resident tensor spans.

use crate::{
    tensor_nbytes, validate_layouts, BindPlan, EngineFacts, Error, ModelFamily, Result,
    ServingRequest, Shape, TensorInventory,
};
use std::ffi::OsString;
use std::sync::Mutex;

#[cfg(test)]
const GIB: u64 = 1 << 30;
const QUANT_BLOCK_ELEMENTS: u64 = 256;
// ds4_glm53_cache_slot: layer/expert u32, last-use/pin epochs u64.
const CACHE_SLOT_BYTES: u64 = 24;
const SELECTED_ID_BYTES: u64 = 4;
const EXPERT_RECENCY_BYTES: u64 = 4;
const PREFILL_ROWS_DEFAULT: u64 = 2048;
const PREFILL_ROWS_MAX: u64 = 2048;
const PREFILL_ROWS_ENV: &str = "DS4_GLM53_PREFILL_ROWS";
const PREFILL_WINDOW: u32 = 4096;
const STAGE_BANKS: u32 = 2;
const PREFETCH_ENV: &str = "DS4_GLM53_PREFETCH";
pub(super) const PREFILL_WINDOW_ENV: &str = "DS4_GLM53_PREFILL_WINDOW";

struct WindowEnv {
    requested: Option<OsString>,
    written: OsString,
}

static WINDOW_ENV: Mutex<Option<WindowEnv>> = Mutex::new(None);

impl WindowEnv {
    fn requested(&self, current: Option<&std::ffi::OsStr>) -> Option<OsString> {
        if current == Some(self.written.as_os_str()) {
            return self.requested.clone();
        }
        current.map(OsString::from)
    }
}

fn window_enabled() -> bool {
    let state = WINDOW_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let current = std::env::var_os(PREFILL_WINDOW_ENV);
    let requested = state
        .as_ref()
        .map(|s| s.requested(current.as_deref()))
        .unwrap_or(current);
    requested
        .as_ref()
        .and_then(|v| v.to_str())
        .is_none_or(|v| v == "4096")
}

pub(super) fn apply_window(window: Option<u32>) {
    let mut state = WINDOW_ENV.lock().unwrap_or_else(|p| p.into_inner());
    let current = std::env::var_os(PREFILL_WINDOW_ENV);
    let requested = state
        .as_ref()
        .map(|s| s.requested(current.as_deref()))
        .unwrap_or_else(|| current.clone());
    let written = OsString::from(window.unwrap_or(0).to_string());
    if current.as_ref() != Some(&written) {
        std::env::set_var(PREFILL_WINDOW_ENV, &written);
    }
    // Native needs the fitted value, but a bank plan's internal 0 must not
    // become a user kill switch for later serial/media allocations. Retain
    // provenance across host threads; an external env change replaces it.
    *state = Some(WindowEnv { requested, written });
}

pub(super) fn prefill_window(
    req: &ServingRequest,
    slots: Option<u32>,
    rows: u32,
    experts: u32,
) -> Option<u32> {
    let serial = matches!(req.max_seqs, crate::MaxSeqs::Off | crate::MaxSeqs::Fixed(1));
    (req.ssd_streaming
        && serial
        && rows >= 128
        && req.ctx >= PREFILL_WINDOW as i32
        && slots.is_some_and(|n| n >= STAGE_BANKS * experts)
        && window_enabled())
    .then_some(PREFILL_WINDOW)
}

pub(super) fn window_bytes(
    req: &ServingRequest,
    shape: Shape,
    rows: u32,
    slots: Option<u32>,
) -> u64 {
    let slots =
        if req.ssd_streaming_cache_experts.is_none() && req.ssd_streaming_cache_bytes.is_none() {
            Some(STAGE_BANKS * shape.n_expert)
        } else {
            slots
        };
    u64::from(prefill_window(req, slots, rows, shape.n_expert).unwrap_or(0))
        * 2
        * u64::from(shape.n_hc)
        * u64::from(shape.n_embd)
        * 4
}

fn staging_banks(req: &ServingRequest, slots: u32, rows: u32, experts: u32) -> u64 {
    let ahead = std::env::var(PREFETCH_ENV).ok().is_none_or(|v| v != "0")
        && prefill_window(req, Some(slots), rows, experts).is_some();
    if ahead {
        u64::from(STAGE_BANKS)
    } else {
        1
    }
}

pub(super) fn prefill_rows(req: &ServingRequest, slots: Option<u32>, used: u32) -> Result<u32> {
    let value = req
        .native_chunk
        .map(|n| n.to_string())
        .or_else(|| std::env::var(PREFILL_ROWS_ENV).ok());
    row_cap(req.ctx, slots.map(u64::from), used, value.as_deref()).map(|rows| rows as u32)
}

fn invalid(message: impl Into<String>) -> Error {
    Error {
        code: 1,
        message: message.into(),
    }
}

/// Retain one metadata quote for preflight and post-open memory accounting.
pub fn probe_ssd_quote(
    facts: &mut EngineFacts,
    req: &ServingRequest,
    shape: Shape,
    inventory: &TensorInventory,
) -> Result<()> {
    if shape.family != ModelFamily::Glm53 || inventory.shards.len() != 1 {
        return Err(invalid("SSD streaming requires one GLM-5.3 GGUF shard"));
    }
    let plan = BindPlan::resolve(shape, inventory);
    plan.check()
        .map_err(|error| invalid(format!("SSD bind: {error}")))?;
    validate_layouts(&plan).map_err(|error| invalid(format!("SSD layout: {error}")))?;

    let mut spans = Vec::new();
    let mut gate_max = 0;
    let mut down_max = 0;
    let mut gate_align = 1;
    let mut down_align = 1;
    for slot in &plan.slots {
        let tensor = slot
            .tensor
            .as_ref()
            .ok_or_else(|| invalid("SSD tensor missing"))?;
        let end = tensor
            .abs_offset
            .checked_add(tensor.bytes)
            .ok_or_else(|| invalid("SSD tensor range overflow"))?;
        if tensor.shard != 0 || end > inventory.shards[0].size {
            return Err(invalid("SSD tensor range exceeds its shard"));
        }
        let routed = slot.name.ends_with("_exps.weight");
        if !routed {
            spans.push((tensor.abs_offset, end));
            continue;
        }
        if tensor.bytes % u64::from(shape.n_expert) != 0 {
            return Err(invalid("SSD expert stride is not integral"));
        }
        let bytes = tensor.bytes / u64::from(shape.n_expert);
        let block = tensor_nbytes(tensor.typ, QUANT_BLOCK_ELEMENTS)
            .ok_or_else(|| invalid("SSD expert type has no byte size"))?;
        if slot.name.ends_with("ffn_down_exps.weight") {
            down_max = down_max.max(bytes);
            down_align = lcm(down_align, block)?;
        } else {
            gate_max = gate_max.max(bytes);
            gate_align = lcm(gate_align, block)?;
        }
    }
    let gate_stride = round_up(gate_max, gate_align)?;
    let down_stride = round_up(down_max, down_align)?;
    let per_slot = gate_stride
        .checked_mul(2)
        .and_then(|gate| gate.checked_add(down_stride))
        .filter(|bytes| *bytes != 0)
        .ok_or_else(|| invalid("SSD cache stride overflow"))?;
    let all = u64::from(shape.n_layer - shape.n_leading_dense) * u64::from(shape.n_expert);
    let capacity = match req.ssd_streaming_cache_experts {
        Some(count) => u64::from(count),
        None => req
            .ssd_streaming_cache_bytes
            .map(|bytes| (bytes / per_slot).min(all))
            .unwrap_or(u64::from(shape.n_expert_used)),
    };
    if capacity < u64::from(shape.n_expert_used) || capacity > all {
        return Err(invalid(format!(
            "SSD cache must hold {}..{all} global experts",
            shape.n_expert_used
        )));
    }
    let cache = capacity
        .checked_mul(per_slot)
        .ok_or_else(|| invalid("SSD cache budget overflow"))?;
    let rows = prefill_rows(req, Some(capacity as u32), shape.n_expert_used)?;
    let selection = u64::from(rows) * u64::from(shape.n_expert_used) * SELECTED_ID_BYTES * 2;
    let metadata = capacity
        .checked_mul(CACHE_SLOT_BYTES)
        .and_then(|bytes| bytes.checked_add(selection))
        .and_then(|bytes| bytes.checked_add(u64::from(shape.n_layer) * u64::from(shape.n_expert)))
        .and_then(|bytes| bytes.checked_add(u64::from(shape.n_expert) * EXPERT_RECENCY_BYTES))
        .ok_or_else(|| invalid("SSD cache metadata overflow"))?;
    facts.ssd_mandatory_bytes = Some(span_bytes(spans)?);
    facts.ssd_cache_experts = Some(capacity as u32);
    facts.ssd_cache_bytes = Some(cache);
    facts.ssd_staging_bytes =
        Some(gate_max.max(down_max) * staging_banks(req, capacity as u32, rows, shape.n_expert));
    facts.ssd_metadata_bytes = Some(metadata);
    Ok(())
}

/// Spend only the expert remainder after context, banks, optional state and
/// reserve have been priced. Prefill staging shares this expert budget.
pub(super) fn fit_auto_cache(
    facts: &mut EngineFacts,
    req: &ServingRequest,
    caps: crate::ServingCaps,
    shape: Shape,
) {
    if !req.ssd_streaming
        || req.ssd_streaming_cache_experts.is_some()
        || req.ssd_streaming_cache_bytes.is_some()
    {
        return;
    }
    let plan = crate::resolve_plan(req, Some(caps), facts);
    let Some(quote) = plan.quote else {
        return;
    };
    if quote.total > quote.available {
        return;
    }
    let count = u64::from(facts.ssd_cache_experts.unwrap_or(0));
    let Some(slot) = facts.ssd_cache_bytes.and_then(|n| n.checked_div(count)) else {
        return;
    };
    let Ok(rows) = prefill_rows(req, Some(count as u32), shape.n_expert_used) else {
        return;
    };
    let rows = plan.effective.native_chunk.unwrap_or(rows);
    let staging = facts.ssd_staging_bytes.unwrap_or(0);
    let transfer = staging / staging_banks(req, count as u32, rows, shape.n_expert);
    let metadata = count * CACHE_SLOT_BYTES;
    let budget = quote.available - quote.total + quote.expert_cache + metadata + staging;
    let all = u64::from(shape.n_layer - shape.n_leading_dense) * u64::from(shape.n_expert);
    let cost = slot + CACHE_SLOT_BYTES;
    let mut capacity = (budget.saturating_sub(transfer) / cost).min(all);
    if staging_banks(req, capacity as u32, rows, shape.n_expert) > 1 {
        // Crossing the two-layer slot threshold funds a second read buffer.
        // If it cannot fit, the largest selected-expert cache stays below it.
        let selected_max = u64::from(STAGE_BANKS * shape.n_expert - 1);
        capacity = (budget.saturating_sub(u64::from(STAGE_BANKS) * transfer) / cost)
            .min(all)
            .max(selected_max);
    }
    facts.ssd_cache_experts = Some(capacity as u32);
    facts.ssd_cache_bytes = Some(capacity * slot);
    facts.ssd_staging_bytes =
        Some(transfer * staging_banks(req, capacity as u32, rows, shape.n_expert));
    facts.ssd_metadata_bytes =
        Some(facts.ssd_metadata_bytes.unwrap_or(0) - metadata + capacity * CACHE_SLOT_BYTES);
}

#[cfg(test)]
fn selection_bytes(ctx: i32, capacity: u64, used: u32, value: Option<&str>) -> Result<u64> {
    Ok(row_cap(ctx, Some(capacity), used, value)? * u64::from(used) * SELECTED_ID_BYTES * 2)
}

#[cfg(test)]
#[test]
fn default_rows_use_capacity() {
    assert_eq!(row_cap(32768, None, 8, None).unwrap(), 2048);
    assert_eq!(row_cap(32768, Some(8), 8, None).unwrap(), 2048);
}

fn row_cap(ctx: i32, slots: Option<u64>, used: u32, value: Option<&str>) -> Result<u64> {
    if ctx <= 0 || used == 0 {
        return Err(invalid(
            "SSD selection requires positive context and expert count",
        ));
    }
    let rows = match value.filter(|value| !value.is_empty()) {
        None => PREFILL_ROWS_DEFAULT,
        Some(value) => match value.parse::<u64>() {
            Ok(rows) if (1..=PREFILL_ROWS_MAX).contains(&rows) => rows,
            _ => return Err(invalid("invalid DS4_GLM53_PREFILL_ROWS (use 1..2048)")),
        },
    };
    if slots.is_some_and(|slots| slots < u64::from(used)) {
        return Err(invalid("SSD cache cannot hold one top-k row"));
    }
    // Only the routed launch is split; its measured union determines fit.
    Ok(rows.min(ctx as u64))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn lcm(a: u64, b: u64) -> Result<u64> {
    (a / gcd(a, b))
        .checked_mul(b)
        .ok_or_else(|| invalid("SSD block alignment overflow"))
}

fn round_up(bytes: u64, alignment: u64) -> Result<u64> {
    let blocks = bytes / alignment + u64::from(bytes % alignment != 0);
    blocks
        .checked_mul(alignment)
        .ok_or_else(|| invalid("SSD stride alignment overflow"))
}

fn span_bytes(mut spans: Vec<(u64, u64)>) -> Result<u64> {
    spans.sort_unstable();
    let mut total = 0u64;
    let mut frontier = 0;
    for (start, end) in spans {
        if end <= frontier {
            continue;
        }
        total = total
            .checked_add(end - start.max(frontier))
            .ok_or_else(|| invalid("SSD resident span overflow"))?;
        frontier = end;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensors::ShardPlan;
    use crate::{expected_layouts, shape_for_variant, TensorInfo, TypeClass, Variant};

    fn inventory() -> TensorInventory {
        let shape = shape_for_variant(Variant::Glm53Flash);
        let mut offset = 0;
        let tensors = expected_layouts(&shape)
            .into_iter()
            .map(|spec| {
                let typ = if spec.name.ends_with("_exps.weight") {
                    let layer: u32 = spec.name.split('.').nth(1).unwrap().parse().unwrap();
                    let edge = [3, 4, 5, 43, 44, 45].contains(&layer);
                    let down = spec.name.ends_with("ffn_down_exps.weight");
                    match (edge, down) {
                        (true, true) => 10,
                        (true, false) | (false, true) => 17,
                        (false, false) => 16,
                    }
                } else {
                    match spec.class {
                        TypeClass::Exact(typ) | TypeClass::OptionalExact(typ) => typ,
                        _ => 8,
                    }
                };
                let elements = spec.dim[..spec.ndim as usize].iter().product();
                let bytes = tensor_nbytes(typ, elements).unwrap();
                let tensor = TensorInfo {
                    name: spec.name,
                    ndim: spec.ndim,
                    dim: spec.dim,
                    typ,
                    rel_offset: offset,
                    abs_offset: offset,
                    elements,
                    bytes,
                    shard: 0,
                };
                offset += bytes;
                tensor
            })
            .collect();
        TensorInventory {
            shards: vec![ShardPlan {
                path: "fixture.gguf".into(),
                size: offset,
                base: 0,
            }],
            tensors,
            data_pos: 0,
            alignment: 32,
            page: 4096,
        }
    }

    #[test]
    fn selected_recipe_quote() {
        let inventory = inventory();
        let mut facts = EngineFacts::default();
        let req = ServingRequest {
            ssd_streaming: true,
            ssd_streaming_cache_bytes: Some(24 * GIB),
            ..ServingRequest::default()
        };
        probe_ssd_quote(
            &mut facts,
            &req,
            shape_for_variant(Variant::Glm53Flash),
            &inventory,
        )
        .unwrap();
        // Native padding: gate LCM(66,74)=2442; down LCM(74,84)=3108.
        assert_eq!(facts.ssd_cache_experts, Some(3389));
        assert_eq!(facts.ssd_cache_bytes, Some(25_768_261_500));
        assert_eq!(facts.ssd_staging_bytes, Some(2_752_512));
        assert_eq!(
            facts.ssd_metadata_bytes,
            Some(
                212_408
                    + (u64::from(crate::SHAPE_GLM53_FLASH.n_layer) + EXPERT_RECENCY_BYTES)
                        * u64::from(crate::SHAPE_GLM53_FLASH.n_expert)
            )
        );
        let mandatory: u64 = inventory
            .tensors
            .iter()
            .filter(|tensor| !tensor.name.ends_with("_exps.weight"))
            .map(|tensor| tensor.bytes)
            .sum();
        assert_eq!(facts.ssd_mandatory_bytes, Some(mandatory));
        assert!(mandatory + facts.ssd_cache_bytes.unwrap() < inventory.shards[0].size);
    }

    #[test]
    fn capacity_bounds_and_count() {
        let inventory = inventory();
        let shape = shape_for_variant(Variant::Glm53Flash);
        let mut facts = EngineFacts::default();
        let req = ServingRequest {
            ssd_streaming: true,
            ssd_streaming_cache_experts: Some(8),
            ..ServingRequest::default()
        };
        probe_ssd_quote(&mut facts, &req, shape, &inventory).unwrap();
        assert_eq!(facts.ssd_cache_bytes, Some(60_828_000));
        assert_eq!(
            facts.ssd_metadata_bytes,
            Some(
                131_264
                    + (u64::from(crate::SHAPE_GLM53_FLASH.n_layer) + EXPERT_RECENCY_BYTES)
                        * u64::from(crate::SHAPE_GLM53_FLASH.n_expert)
            )
        );
        for count in [0, 7, 12385] {
            let req = ServingRequest {
                ssd_streaming_cache_experts: Some(count),
                ..req.clone()
            };
            assert!(probe_ssd_quote(&mut facts, &req, shape, &inventory).is_err());
        }
    }

    #[test]
    fn resident_ranges_are_unioned() {
        assert_eq!(
            span_bytes(vec![(0, 10), (5, 15), (20, 30), (21, 25)]).unwrap(),
            25
        );
        let mut inventory = inventory();
        inventory.tensors[0].abs_offset = u64::MAX;
        let req = ServingRequest {
            ssd_streaming: true,
            ..ServingRequest::default()
        };
        assert!(probe_ssd_quote(
            &mut EngineFacts::default(),
            &req,
            shape_for_variant(Variant::Glm53Flash),
            &inventory
        )
        .is_err());
    }

    #[test]
    fn selection_batch_budget() {
        assert_eq!(selection_bytes(2048, 3389, 8, None).unwrap(), 131072);
        assert_eq!(selection_bytes(2048, 8, 8, None).unwrap(), 131072);
        assert_eq!(selection_bytes(2, 3389, 8, None).unwrap(), 128);
        for (value, rows) in [("1", 1), ("128", 128), ("256", 256)] {
            assert_eq!(
                selection_bytes(2048, 3389, 8, Some(value)).unwrap(),
                rows * 64
            );
        }
        assert_eq!(selection_bytes(2048, 64, 8, Some("256")).unwrap(), 16384);
        for value in ["0", "2049", "-1", "invalid"] {
            assert!(selection_bytes(2048, 3389, 8, Some(value)).is_err());
        }
    }

    #[test]
    fn wide_rows_keep_small_cache() {
        let req = ServingRequest {
            ctx: 16384,
            native_chunk: Some(2048),
            ..ServingRequest::default()
        };
        assert_eq!(prefill_rows(&req, Some(8), 8).unwrap(), 2048);
        assert!(selection_bytes(16384, 8, 8, Some("2049")).is_err());
    }

    #[test]
    fn auto_cache_prices_state_first() {
        let shape = shape_for_variant(Variant::Glm53Flash);
        let caps = crate::caps_from_shape(shape);
        let inventory = inventory();
        let mut capacity = Vec::new();
        for ctx in [131072, 262144] {
            let req = ServingRequest {
                ctx,
                max_seqs: crate::MaxSeqs::Fixed(2),
                native_chunk: Some(2048),
                mtp_mode: crate::MtpMode::On,
                mtp_draft: Some(3),
                prefix_reuse: crate::PrefixReuse::Partial,
                ssd_streaming: true,
                ..ServingRequest::default()
            };
            let mut facts = EngineFacts::default();
            probe_ssd_quote(&mut facts, &req, shape, &inventory).unwrap();
            assert_eq!(facts.ssd_cache_experts, Some(shape.n_expert_used));
            let host = crate::QuoteHost {
                weights_bytes: facts.ssd_mandatory_bytes.unwrap(),
                mtp_bytes: 0,
                available_bytes: 90 * GIB,
                native_chunk: None,
                vision: true,
            };
            crate::fill_quote_facts(&mut facts, &req, caps, Some(shape), host);
            fit_auto_cache(&mut facts, &req, caps, shape);
            let plan = crate::resolve_plan(&req, Some(caps), &facts);
            let quote = plan.quote.unwrap();
            assert!(quote.total <= quote.available);
            assert_eq!(plan.effective.ctx, ctx);
            assert_eq!(plan.effective.max_seqs, 2);
            assert_eq!(plan.effective.native_chunk, Some(2048));
            assert!(quote.floor >= 12 * GIB);
            assert!(quote.checkpoint_pool > 0 && quote.media_reserve > 0 && quote.mtp_state > 0);
            capacity.push(plan.effective.ssd_streaming_cache_experts.unwrap());
        }
        assert!(capacity[1] < capacity[0]);
    }

    #[test]
    fn glm_review_auto_staging() {
        let shape = shape_for_variant(Variant::Glm53Flash);
        let caps = crate::caps_from_shape(shape);
        let inventory = inventory();
        let req = ServingRequest {
            ctx: 32768,
            max_seqs: crate::MaxSeqs::Fixed(1),
            native_chunk: Some(2048),
            ssd_streaming: true,
            ..ServingRequest::default()
        };
        let transfer = inventory
            .tensors
            .iter()
            .filter(|t| t.name.ends_with("_exps.weight"))
            .map(|t| t.bytes / u64::from(shape.n_expert))
            .max()
            .unwrap();
        let threshold = STAGE_BANKS * shape.n_expert;
        for (target, stage, shortfall, rows) in [
            (threshold - 1, 1, 0, 2048),
            (threshold, 2, 0, 2048),
            (threshold + 1, 2, 0, 2048),
            (threshold, 2, transfer, 2048),
            (threshold, 1, 0, 64),
        ] {
            let mut facts = EngineFacts::default();
            probe_ssd_quote(&mut facts, &req, shape, &inventory).unwrap();
            assert_eq!(facts.ssd_staging_bytes, Some(transfer));
            let count = u64::from(shape.n_expert_used);
            let slot = facts.ssd_cache_bytes.unwrap() / count;
            let host = crate::QuoteHost {
                weights_bytes: facts.ssd_mandatory_bytes.unwrap(),
                mtp_bytes: 0,
                available_bytes: 90 * GIB,
                native_chunk: None,
                vision: false,
            };
            let fitted = ServingRequest {
                native_chunk: Some(rows),
                ..req.clone()
            };
            crate::fill_quote_facts(&mut facts, &fitted, caps, Some(shape), host);
            let total = crate::resolve_plan(&req, Some(caps), &facts)
                .quote
                .unwrap()
                .total;
            let fixed = total - count * (slot + CACHE_SLOT_BYTES) - transfer;
            facts.host_available_bytes = Some(
                fixed + u64::from(target) * (slot + CACHE_SLOT_BYTES) + stage * transfer
                    - shortfall,
            );
            fit_auto_cache(&mut facts, &req, caps, shape);
            let capacity = if shortfall == 0 { target } else { target - 1 };
            let banks = if capacity < threshold || rows < 128 {
                1
            } else {
                2
            };
            assert_eq!(facts.ssd_cache_experts, Some(capacity));
            assert_eq!(facts.ssd_staging_bytes, Some(banks * transfer));
            let quote = crate::resolve_plan(&req, Some(caps), &facts).quote.unwrap();
            assert!(quote.total <= quote.available);

            // Repricing the fitted count must match an explicit native cache.
            let pinned = ServingRequest {
                ssd_streaming_cache_experts: Some(capacity),
                ..fitted
            };
            let mut exact = EngineFacts::default();
            probe_ssd_quote(&mut exact, &pinned, shape, &inventory).unwrap();
            assert_eq!(facts.ssd_staging_bytes, exact.ssd_staging_bytes);
            fit_auto_cache(&mut facts, &req, caps, shape);
            assert_eq!(facts.ssd_cache_experts, Some(capacity));
            assert_eq!(facts.ssd_staging_bytes, exact.ssd_staging_bytes);
        }
    }
}
