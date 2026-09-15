//! Model-free wiring gates for the Ling-3.0-flash-VL family contract.

use ds4_core::{
    bind_names, expected_layouts, route_architecture, serving_caps, shape_for_variant, ArchRoute,
    ModelFamily, Support, Variant,
};

fn shape() -> ds4_core::Shape {
    shape_for_variant(Variant::Ling30FlashVl)
}

#[test]
fn architecture_routes_to_the_family() {
    assert!(matches!(
        route_architecture(Some(b"bailingmoe3")),
        ArchRoute::Fixed(Variant::Ling30FlashVl)
    ));
    let shape = shape();
    assert_eq!(shape.family, ModelFamily::Ling3Vl);
    assert_eq!(shape.n_layer, 42);
    assert_eq!(shape.n_expert, 512);
    assert_eq!(shape.n_expert_used, 8);
    assert_eq!(shape.n_kv_lora, 512);
    assert_eq!(shape.n_key_mla, 192);
    assert_eq!(shape.n_kda_head_dim, 128);
    assert_eq!(shape.n_full_attn_count, 7);
}

#[test]
fn bind_plan_requires_every_published_tensor() {
    let names = bind_names(&shape());
    assert_eq!(names.len(), 917);
    // Both attention families and both FFN families must be represented.
    for needle in [
        "blk.0.ffn_gate.weight",
        "blk.4.ssm_norm.weight",
        "blk.5.attn_k_b.weight",
        "blk.5.attn_gate.weight",
        "blk.41.ffn_down_exps.weight",
        "output.weight",
    ] {
        assert!(
            names.iter().any(|n| n.name == needle),
            "bind plan is missing {needle}"
        );
    }
    // A KDA block has no MLA latent projection and vice versa.
    assert!(!names.iter().any(|n| n.name == "blk.4.attn_k_b.weight"));
    assert!(!names.iter().any(|n| n.name == "blk.5.ssm_norm.weight"));
    assert_eq!(expected_layouts(&shape()).len(), names.len());
}

#[test]
fn serving_matches_the_qwen_surface_it_was_sized_against() {
    let caps = serving_caps(ModelFamily::Ling3Vl, Variant::Ling30FlashVl);
    let qwen = serving_caps(ModelFamily::Qwen4Exp, Variant::Qwen38FlashNext);
    assert_eq!(caps.banks, qwen.banks);
    assert_eq!(caps.bank_support, qwen.bank_support);
    assert_eq!(caps.reuse, qwen.reuse);
    assert_eq!(caps.reuse_support, qwen.reuse_support);
    assert_eq!(caps.disk, qwen.disk);
    assert_eq!(caps.snapshot, qwen.snapshot);
    assert_eq!(caps.qualified_banks, Some(2));
    // This architecture has no NextN predictor, so no draft lane exists.
    assert_eq!(caps.mtp_support, Support::None);
}
