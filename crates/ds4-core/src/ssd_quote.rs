//! Mirror the native GLM cache padding and validated resident tensor spans.

use crate::{
    tensor_nbytes, validate_layouts, BindPlan, EngineFacts, Error, ModelFamily, Result,
    ServingRequest, Shape, TensorInventory,
};

const GIB: u64 = 1 << 30;
const DEFAULT_CACHE_BYTES: u64 = 24 * GIB;
const QUANT_BLOCK_ELEMENTS: u64 = 256;
// ds4_glm53_cache_slot: layer/expert u32, last-use/pin epochs u64.
const CACHE_SLOT_BYTES: u64 = 24;
const SELECTED_ID_BYTES: u64 = 4;
const PREFILL_ROWS_DEFAULT: u64 = 128;
const PREFILL_ROWS_MAX: u64 = 256;
const PREFILL_ROWS_ENV: &str = "DS4_GLM53_PREFILL_ROWS";

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
        None => (req.ssd_streaming_cache_bytes.unwrap_or(DEFAULT_CACHE_BYTES) / per_slot).min(all),
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
    let selection = u64::from(prefill_rows(
        req,
        Some(capacity as u32),
        shape.n_expert_used,
    )?) * u64::from(shape.n_expert_used)
        * SELECTED_ID_BYTES
        * 2;
    let metadata = capacity
        .checked_mul(CACHE_SLOT_BYTES)
        .and_then(|bytes| bytes.checked_add(selection))
        .ok_or_else(|| invalid("SSD cache metadata overflow"))?;
    facts.ssd_mandatory_bytes = Some(span_bytes(spans)?);
    facts.ssd_cache_experts = Some(capacity as u32);
    facts.ssd_cache_bytes = Some(cache);
    facts.ssd_staging_bytes = Some(gate_max.max(down_max));
    facts.ssd_metadata_bytes = Some(metadata);
    Ok(())
}

#[cfg(test)]
fn selection_bytes(ctx: i32, capacity: u64, used: u32, value: Option<&str>) -> Result<u64> {
    Ok(row_cap(ctx, Some(capacity), used, value)? * u64::from(used) * SELECTED_ID_BYTES * 2)
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
            Ok(rows @ (1 | PREFILL_ROWS_DEFAULT | PREFILL_ROWS_MAX)) => rows,
            _ => return Err(invalid("invalid DS4_GLM53_PREFILL_ROWS (use 1/128/256)")),
        },
    };
    // Native subdivides each FFN launch so all selected experts fit pinned.
    let rows = rows.min(ctx as u64);
    Ok(slots
        .map(|slots| rows.min(slots / u64::from(used)))
        .unwrap_or(rows))
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
        assert_eq!(facts.ssd_metadata_bytes, Some(89_528));
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
        assert_eq!(facts.ssd_metadata_bytes, Some(256));
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
        assert_eq!(selection_bytes(2048, 3389, 8, None).unwrap(), 8192);
        assert_eq!(selection_bytes(2048, 8, 8, None).unwrap(), 64);
        assert_eq!(selection_bytes(2, 3389, 8, None).unwrap(), 128);
        for (value, rows) in [("1", 1), ("128", 128), ("256", 256)] {
            assert_eq!(
                selection_bytes(2048, 3389, 8, Some(value)).unwrap(),
                rows * 64
            );
        }
        assert_eq!(selection_bytes(2048, 64, 8, Some("256")).unwrap(), 512);
        for value in ["0", "129", "-1", "invalid"] {
            assert!(selection_bytes(2048, 3389, 8, Some(value)).is_err());
        }
    }
}
