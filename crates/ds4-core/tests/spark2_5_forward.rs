//! Spark-X2.6-4B VQ: shape contract and fixture sanity. The live forward
//! gate needs the release and a CUDA device, so it is the CLI run recorded in
//! docs/spark-x2.6-vq.md rather than a model-free crate test.
//! The reference fixture is the checkpoint in F32 on CPU. The release's
//! matrices are VQ-encoded (artifact/manifest.txt: mean per-matrix relL2 about
//! 23%), so the pair is gated on the argmax and on the top-k set, never on a
//! tight logit distance: the exact walk against dequantized weights lives in
//! docs/spark-x2.6-vq.md, not in a model-free crate test.

use ds4_core::{shape_for_variant, ModelFamily, Variant};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/spark2_5/logits-reference.json"
    ))
    .expect("logits fixture")
}

fn dequantized_fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/spark2_5/logits-reference-dequantized.json"
    ))
    .expect("dequantized logits fixture")
}

#[test]
fn shape_matches_the_published_config() {
    let shape = shape_for_variant(Variant::SparkX26_4bVq);
    assert_eq!(shape.family, ModelFamily::Spark2_5Vq);
    assert_eq!(shape.name, "Spark-X2.6-4B-VQ");
    assert_eq!(
        (shape.n_layer, shape.n_embd, shape.n_vocab),
        (36, 2_560, 131_072)
    );
    assert_eq!(
        (shape.n_head, shape.n_head_kv, shape.n_head_dim),
        (16, 4, 256)
    );
    assert_eq!(shape.n_ff_dense, 10_240);
    assert_eq!(shape.n_swa, 512);
    assert_eq!(shape.n_full_attn_count, 9);
}

#[test]
fn caps_stay_conservative() {
    let caps = ds4_core::caps_from_shape(shape_for_variant(Variant::SparkX26_4bVq));
    // The serial forward, the exact-prefix rewind, the session payload and the
    // banked lane are implemented but not yet gated, so they are Present,
    // never Qualified. A drafter does not exist.
    assert_eq!(caps.banks, ds4_core::BankLane::Persistent);
    assert_eq!(caps.bank_support, ds4_core::Support::Present);
    assert_eq!(caps.reuse, ds4_core::ReuseKind::Partial);
    assert_eq!(caps.reuse_support, ds4_core::Support::Present);
    assert_eq!(caps.disk, ds4_core::Support::Present);
    assert_eq!(caps.snapshot, ds4_core::Support::Present);
    assert_eq!(caps.mtp, ds4_core::MtpKind::None);
    assert_eq!(caps.qualified_ctx, None);
    assert_eq!(caps.qualified_banks, None);
}

#[test]
fn the_two_references_agree_on_the_prompt_and_the_argmax() {
    let checkpoint = fixture();
    let dequantized = dequantized_fixture();
    assert_eq!(checkpoint["token_ids"], dequantized["token_ids"]);
    assert_eq!(checkpoint["argmax"], dequantized["argmax"]);
    // Same leading id, different logit scales: the checkpoint's matrices are
    // not the artifact's weights, so only the ordering is comparable there.
    let cp = checkpoint["top_k"][0]["logit"].as_f64().unwrap();
    let dq = dequantized["top_k"][0]["logit"].as_f64().unwrap();
    assert!((cp - dq).abs() > 1.0, "the two references should differ");
}

#[test]
fn fixture_is_self_consistent() {
    let f = fixture();
    let ids: Vec<i64> = f["token_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(ids, [1046, 11862, 314, 17349, 390]);
    let top = f["top_k"].as_array().expect("top_k");
    assert_eq!(top.len(), 64);
    assert_eq!(
        top[0]["id"].as_i64().unwrap(),
        f["argmax"].as_i64().unwrap()
    );
    let mut last = f64::INFINITY;
    for entry in top {
        let value = entry["logit"].as_f64().unwrap();
        assert!(value <= last, "top_k is not ordered");
        last = value;
    }
}
