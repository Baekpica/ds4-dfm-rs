//! Model-free loader/planner gates; no native execution qualification.

use ds4_core::{
    expected_layouts, shape_for_variant, tensor_nbytes, validate_layouts, BindNeed, BindPlan,
    BindSlot, TensorInfo, TypeClass, Variant,
};

const IQ2_XXS: u32 = 16;
const IQ2_XS: u32 = 17;
const Q2_K: u32 = 10;
const EDGES: [u32; 6] = [3, 4, 5, 43, 44, 45];

fn fixture(variant: Variant) -> BindPlan {
    let shape = shape_for_variant(variant);
    let slots = expected_layouts(&shape)
        .into_iter()
        .map(|spec| {
            let layer = spec
                .name
                .split('.')
                .nth(1)
                .and_then(|s| s.parse::<u32>().ok());
            let routed = spec.name.contains("_exps.weight");
            let typ = if routed && variant == Variant::Glm53Flash {
                let edge = EDGES.contains(&layer.unwrap());
                match (edge, spec.name.contains("ffn_down_exps")) {
                    (true, true) => Q2_K,
                    (true, false) | (false, true) => IQ2_XS,
                    (false, false) => IQ2_XXS,
                }
            } else {
                match spec.class {
                    TypeClass::Exact(t) | TypeClass::OptionalExact(t) => t,
                    TypeClass::Plain => 1,
                    TypeClass::Routed => IQ2_XXS,
                    _ => 8,
                }
            };
            BindSlot {
                name: spec.name.clone(),
                need: BindNeed::Required,
                tensor: Some(TensorInfo {
                    name: spec.name,
                    ndim: spec.ndim,
                    dim: spec.dim,
                    typ,
                    rel_offset: 0,
                    abs_offset: 0,
                    elements: 0,
                    bytes: 0,
                    shard: 0,
                }),
                index: Some(0),
            }
        })
        .collect();
    BindPlan {
        shape,
        slots,
        n_shards: 1,
        data_pos: 0,
        alignment: 32,
        page: 4096,
    }
}

#[test]
fn selected_boundary6_layout() {
    let plan = fixture(Variant::Glm53Flash);
    assert_eq!((plan.shape.n_expert, plan.shape.n_expert_used), (288, 8));
    validate_layouts(&plan).expect("selected GLM53 metadata layout");
    let count = |typ| {
        plan.slots
            .iter()
            .filter(|slot| {
                slot.name.contains("_exps.weight") && slot.tensor.as_ref().unwrap().typ == typ
            })
            .count()
    };
    assert_eq!((count(IQ2_XXS), count(IQ2_XS), count(Q2_K)), (74, 49, 6));
}

#[test]
fn rejects_mismatched_gate_up() {
    let mut plan = fixture(Variant::Glm53Flash);
    plan.slots
        .iter_mut()
        .find(|slot| slot.name == "blk.3.ffn_up_exps.weight")
        .unwrap()
        .tensor
        .as_mut()
        .unwrap()
        .typ = IQ2_XXS;
    assert_eq!(validate_layouts(&plan).unwrap_err().token(), "gate-up 3");
}

#[test]
fn rejects_wrong_expert_shape() {
    let mut plan = fixture(Variant::Glm53Flash);
    plan.slots
        .iter_mut()
        .find(|slot| slot.name == "blk.3.ffn_gate_exps.weight")
        .unwrap()
        .tensor
        .as_mut()
        .unwrap()
        .dim[2] = 256;
    assert_eq!(
        validate_layouts(&plan).unwrap_err().token(),
        "dim blk.3.ffn_gate_exps.weight"
    );
}

#[test]
fn rejects_below_floor() {
    let mut plan = fixture(Variant::Glm53Flash);
    plan.slots
        .iter_mut()
        .find(|slot| slot.name == "blk.3.ffn_down_exps.weight")
        .unwrap()
        .tensor
        .as_mut()
        .unwrap()
        .typ = 19;
    assert_eq!(
        validate_layouts(&plan).unwrap_err().token(),
        "type blk.3.ffn_down_exps.weight"
    );
}

#[test]
fn deepseek_layout_closed() {
    let mut plan = fixture(Variant::Flash);
    for slot in &mut plan.slots {
        if slot.name.contains("ffn_gate_exps") || slot.name.contains("ffn_up_exps") {
            slot.tensor.as_mut().unwrap().typ = IQ2_XS;
        }
    }
    assert!(validate_layouts(&plan).is_err());
}

#[test]
fn xs_bytes_and_ceiling() {
    assert_eq!(tensor_nbytes(IQ2_XS, 256), Some(74));
    assert_eq!(tensor_nbytes(IQ2_XS, 257), Some(148));
    assert_eq!(tensor_nbytes(IQ2_XS, 4096 * 2048 * 288), Some(698_351_616));
    // Saturating elements+255 loses a block near u64::MAX.
    assert_eq!(
        tensor_nbytes(IQ2_XS, u64::MAX),
        Some(5_332_261_958_806_667_264)
    );
    assert_eq!(tensor_nbytes(0, u64::MAX), None);
}
