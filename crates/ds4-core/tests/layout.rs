//! Host weights_validate_layout vs C layout_c_oracle.

use ds4_core::{
    bind_dspark_names, bind_mtp_names, bind_names, dump_expected_layouts, dump_expected_support,
    dump_layout_check_tapes, expected_dspark_layouts, expected_layouts, expected_mtp_layouts,
    identify_gguf, shape_for_variant, validate_dspark_layouts, validate_layouts,
    validate_mtp_layouts, validate_support_layouts, BindNeed, BindPlan, BindSlot, LayoutSpec,
    SupportCatalog, TensorInfo, TensorInventory, TypeClass, Variant, DSPARK_MARKOV_RANK,
    SHAPE_FLASH,
};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_LAYOUT_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/layout_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/layout_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_out(args: &[&str]) -> String {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run layout_c_oracle");
    assert!(
        out.status.success(),
        "oracle {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

#[test]
fn dump_matches_c_oracle() {
    assert_eq!(c_out(&[]), dump_expected_layouts());
}

#[test]
fn check_tapes_match_c_oracle() {
    assert_eq!(c_out(&["check"]), dump_layout_check_tapes());
}

#[test]
fn support_dump_matches_c_oracle() {
    assert_eq!(c_out(&["support"]), dump_expected_support());
}

fn fake_tensor(name: &str, typ: u32, ndim: u32, dim: [u64; 8]) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        ndim,
        dim,
        typ,
        rel_offset: 0,
        abs_offset: 0,
        elements: 0,
        bytes: 0,
        shard: 0,
    }
}

fn class_type(class: TypeClass, name: &str, routed_up: u32) -> u32 {
    match class {
        TypeClass::Exact(t) | TypeClass::OptionalExact(t) => t,
        TypeClass::Plain => 1,
        TypeClass::Routed => {
            if name.contains("ffn_up_exps") {
                routed_up
            } else {
                16
            }
        }
        _ => 8,
    }
}

fn plan_from_specs(specs: &[LayoutSpec], routed_up: u32) -> BindPlan {
    let slots = specs
        .iter()
        .map(|s| BindSlot {
            name: s.name.clone(),
            need: BindNeed::Required,
            tensor: Some(fake_tensor(
                &s.name,
                class_type(s.class, &s.name, routed_up),
                s.ndim,
                s.dim,
            )),
            index: Some(0),
        })
        .collect();
    BindPlan {
        shape: SHAPE_FLASH,
        slots,
        n_shards: 1,
        data_pos: 0,
        alignment: 32,
        page: 4096,
    }
}

#[test]
fn validate_mtp_accepts_matching_plan() {
    let specs = expected_mtp_layouts(&SHAPE_FLASH);
    let plan = plan_from_specs(&specs, 16);
    validate_mtp_layouts(&plan).expect("mtp layout ok");
}

#[test]
fn validate_mtp_rejects_gate_up_mismatch() {
    let specs = expected_mtp_layouts(&SHAPE_FLASH);
    let plan = plan_from_specs(&specs, 10);
    let err = validate_mtp_layouts(&plan).expect_err("gate/up type mismatch");
    assert_eq!(err.token(), "gate-up 0");
}

#[test]
fn validate_dspark_accepts_matching_plan() {
    let specs = expected_dspark_layouts(&SHAPE_FLASH, DSPARK_MARKOV_RANK);
    let plan = plan_from_specs(&specs, 16);
    validate_dspark_layouts(&plan, DSPARK_MARKOV_RANK).expect("dspark layout ok");
    validate_support_layouts(&plan, Some(SupportCatalog::Dspark)).expect("support dispatch");
}

#[test]
fn support_layout_covers_bind_catalog() {
    for v in [Variant::Flash, Variant::Pro] {
        let shape = shape_for_variant(v);
        let mtp_bind: HashSet<String> = bind_mtp_names().into_iter().map(|n| n.name).collect();
        let mtp_layout: HashSet<String> = expected_mtp_layouts(&shape)
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(mtp_bind, mtp_layout, "{v:?} MTP bind/layout name set");
        let ds_bind: HashSet<String> = bind_dspark_names().into_iter().map(|n| n.name).collect();
        let ds_layout: HashSet<String> = expected_dspark_layouts(&shape, DSPARK_MARKOV_RANK)
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(ds_bind, ds_layout, "{v:?} DSpark bind/layout name set");
    }
}

#[test]
fn layout_covers_bind_catalog() {
    for v in [
        Variant::Flash,
        Variant::Pro,
        Variant::SolarOpen2_250B,
        Variant::Motif3,
        Variant::Kexaone236B,
        Variant::Dots3NotePrev,
        Variant::Qwen38FlashNext,
        Variant::Glm53Flash,
        Variant::K2Horizon375B,
    ] {
        let shape = shape_for_variant(v);
        let bind: HashSet<String> = bind_names(&shape).into_iter().map(|n| n.name).collect();
        let layout: HashSet<String> = expected_layouts(&shape)
            .into_iter()
            .map(|s| s.name)
            .collect();
        let missing: Vec<_> = bind.difference(&layout).cloned().collect();
        let extra: Vec<_> = layout.difference(&bind).cloned().collect();
        assert!(
            missing.is_empty(),
            "{v:?} bind names without layout: {missing:?}"
        );
        assert!(
            extra.is_empty(),
            "{v:?} layout names without bind: {extra:?}"
        );
    }
}

#[test]
fn k2_horizon_layout_is_exactly_842_tensors() {
    let shape = shape_for_variant(Variant::K2Horizon375B);
    let specs = expected_layouts(&shape);
    let names: HashSet<_> = specs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(specs.len(), 842);
    assert!(names.contains("blk.2.ffn_down.weight"));
    assert!(names.contains("blk.3.ffn_gate_exps.weight"));
    assert!(names.contains("blk.60.ffn_down_shexp.weight"));
    assert!(!names.contains("blk.3.attn_q_norm.weight"));
    assert!(!names.contains("blk.60.nextn.eh_proj.weight"));
}

#[test]
fn k2_horizon_layout_accepts_mq87_iq_types() {
    let shape = shape_for_variant(Variant::K2Horizon375B);
    for typ in [17u32, 19u32, 29u32] {
        let specs = expected_layouts(&shape);
        let mut plan = plan_from_specs(&specs, typ);
        plan.shape = shape;
        for slot in &mut plan.slots {
            if slot.name.contains("ffn_gate_exps") || slot.name.contains("ffn_up_exps") {
                slot.tensor.as_mut().unwrap().typ = typ;
            }
            if slot.name.contains("ffn_down_exps") {
                slot.tensor.as_mut().unwrap().typ = if typ == 19 { 16 } else { 17 };
            }
        }
        validate_layouts(&plan).expect("MQ87 routed IQ type accepted");
    }
}

#[test]
fn validate_glm53_artifact_when_configured() {
    let Ok(path) = std::env::var("DS4_GLM53_MODEL") else {
        return;
    };
    let path = std::path::Path::new(&path);
    let shape = identify_gguf(path).expect("identify GLM artifact").shape;
    assert_eq!(shape.variant, Variant::Glm53Flash);
    let inventory = TensorInventory::open(path).expect("inventory GLM artifact");
    let plan = BindPlan::resolve(shape, &inventory);
    plan.check().expect("complete GLM bind plan");
    validate_layouts(&plan).expect("valid GLM tensor layouts");
}

#[test]
fn validate_k2_horizon_artifact_when_configured() {
    let Ok(path) = std::env::var("DS4_K2_MODEL") else {
        return;
    };
    let path = std::path::Path::new(&path);
    let shape = identify_gguf(path).expect("identify K2 artifact").shape;
    assert_eq!(shape.variant, Variant::K2Horizon375B);
    let inventory = TensorInventory::open(path).expect("inventory K2 artifact");
    assert_eq!(inventory.tensors.len(), 842);
    let plan = BindPlan::resolve(shape, &inventory);
    plan.check().expect("complete K2 bind plan");
    validate_layouts(&plan).expect("valid K2 MQ87 tensor layouts");
}
