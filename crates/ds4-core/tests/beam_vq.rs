//! Beam-1.0 VQ release contract. Model-backed checks read only JSON assets
//! and GGUF parts (no GPU); point DS4_BEAM_MODEL at the release directory.
use ds4_core::{identify_gguf, ModelFamily, TensorInventory, Variant, Vocab};
use serde_json::Value;
use std::path::PathBuf;

const PARTS: usize = 51;
const TENSORS: usize = 38_415;

fn root() -> PathBuf {
    PathBuf::from(std::env::var_os("DS4_BEAM_MODEL").expect("DS4_BEAM_MODEL"))
}

fn cases() -> Value {
    serde_json::from_str(include_str!("../../../tests/fixtures/beam/eval-cases.json")).unwrap()
}

#[test]
fn family_is_its_own_contract() {
    let shape = ds4_core::shape_for_variant(Variant::BeamVq);
    assert_eq!(shape.family, ModelFamily::BeamVq);
    assert_eq!(shape.name, "Beam-1.0-501B-A23B-VQ");
    assert_eq!(
        (shape.n_layer, shape.n_embd, shape.n_vocab),
        (52, 5120, 151_936)
    );
    assert_eq!(
        (shape.n_head, shape.n_head_kv, shape.n_head_dim, shape.n_swa),
        (48, 8, 128, 512)
    );
    assert_eq!(
        (shape.n_expert, shape.n_ff_exp, shape.n_leading_dense),
        (256, 2560, 3)
    );
    let caps = ds4_core::caps_from_shape(shape);
    assert_eq!(caps.ctx_max, Some(1_048_576));
    assert_eq!(caps.qualified_ctx, None);
}

#[test]
#[ignore = "requires the release in DS4_BEAM_MODEL; hashes every part, no GPU"]
fn release_opens_every_part() {
    let root = root();
    let id = identify_gguf(&root).unwrap();
    assert_eq!(id.shape.family, ModelFamily::BeamVq);
    assert_eq!(id.split_count as usize, PARTS);
    let inventory = TensorInventory::open(&root).unwrap();
    assert_eq!(inventory.shards.len(), PARTS);
    assert_eq!(inventory.tensors.len(), TENSORS);
    // The embedding part is the release's last; native keeps it on the host.
    let last = inventory.tensors.last().unwrap();
    assert_eq!(last.name, "model.embed_tokens.weight");
    assert_eq!(last.shard as usize, PARTS - 1);
    ds4_core::validate_gguf(&root).unwrap();
    println!("BEAM_RELEASE_PASS parts={PARTS} tensors={TENSORS}");
}

#[test]
#[ignore = "requires the release JSON assets in DS4_BEAM_MODEL; no GPU"]
fn source_tokenizer_matches_the_eval_ids() {
    let vocab = Vocab::load_path(&root(), ModelFamily::BeamVq).unwrap();
    assert_eq!((vocab.eos_id, vocab.eot_id), (151_674, 151_675));
    assert!(vocab.is_stop(151_674) && vocab.is_stop(151_675) && !vocab.is_stop(151_670));
    for case in cases()["cases"].as_array().unwrap() {
        let want: Vec<i32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let got = vocab.encode_rendered_chat(case["text"].as_str().unwrap());
        assert!(got.len() >= want.len(), "{}", case["id"]);
        assert_eq!(&got[..want.len()], &want[..], "{}", case["id"]);
    }
    // The server forces Harmony headers by these ids (ds4-server tools/beam.rs).
    for (text, ids) in [
        (
            "<|channel|>analysis<|message|>",
            &[151_672, 34_484, 151_671][..],
        ),
        (
            "<|channel|>final<|message|>",
            &[151_672, 11_822, 151_671][..],
        ),
        ("<|end|><|start|>assistant", &[151_670, 151_669, 77_091][..]),
        (
            "<|channel|>analysis <|constrain|>xml<|message|>",
            &[151_672, 34_484, 220, 151_673, 6_455, 151_671][..],
        ),
    ] {
        assert_eq!(vocab.encode_rendered_chat(text), ids, "{text}");
    }
}
