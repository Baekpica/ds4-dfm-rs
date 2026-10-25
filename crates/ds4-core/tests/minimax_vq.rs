//! MiniMax-M3.1-Flash VQ release contract. Model-backed checks read only
//! JSON assets and GGUF parts (no GPU); point DS4_MINIMAX_MODEL at the
//! release directory.
use ds4_core::{identify_gguf, ModelFamily, TensorInventory, Variant, Vocab};
use serde_json::Value;
use std::path::PathBuf;

// release.json: 57 expert parts, the body and the embedding part (the
// vision and drafter parts stay out of the trunk inventory).
const PARTS: usize = 59;
const TENSORS: usize = 30_662;
const DRAFT_TENSORS: usize = 62;

fn root() -> PathBuf {
    PathBuf::from(std::env::var_os("DS4_MINIMAX_MODEL").expect("DS4_MINIMAX_MODEL"))
}

fn cases() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/minimax/eval-cases.json"
    ))
    .unwrap()
}

#[test]
fn family_is_its_own_contract() {
    let shape = ds4_core::shape_for_variant(Variant::MinimaxVq);
    assert_eq!(shape.family, ModelFamily::MinimaxVq);
    assert_eq!(shape.name, "MiniMax-M3.1-Flash-VQ");
    assert_eq!(
        (shape.n_layer, shape.n_embd, shape.n_vocab),
        (60, 6144, 200_064)
    );
    assert_eq!(
        (shape.n_head, shape.n_head_kv, shape.n_head_dim, shape.n_rot),
        (64, 4, 128, 64)
    );
    assert_eq!(
        (
            shape.n_expert,
            shape.n_expert_used,
            shape.n_ff_exp,
            shape.n_leading_dense
        ),
        (128, 4, 3072, 3)
    );
    let caps = ds4_core::caps_from_shape(shape);
    assert_eq!(caps.ctx_max, Some(1_048_576));
    assert_eq!(caps.qualified_ctx, None);
}

#[test]
#[ignore = "requires the release in DS4_MINIMAX_MODEL; hashes every part, no GPU"]
fn release_opens_every_part() {
    let root = root();
    let id = identify_gguf(&root).unwrap();
    assert_eq!(id.shape.family, ModelFamily::MinimaxVq);
    assert_eq!(id.split_count as usize, PARTS);
    let inventory = TensorInventory::open(&root).unwrap();
    assert_eq!(inventory.shards.len(), PARTS);
    assert_eq!(inventory.tensors.len(), TENSORS);
    // The embedding part is the inventory's last; native keeps it on the host.
    let last = inventory.tensors.last().unwrap();
    assert_eq!(last.name, "language_model.model.embed_tokens.weight");
    assert_eq!(last.shard as usize, PARTS - 1);
    ds4_core::validate_gguf(&root).unwrap();
    // Drafting maps the DSpark part before the embedding part.
    let draft = TensorInventory::open_minimax(&root, ds4_core::MinimaxFeature::Draft).unwrap();
    assert_eq!(draft.shards.len(), PARTS + 1);
    assert_eq!(draft.tensors.len(), TENSORS + DRAFT_TENSORS);
    assert_eq!(
        draft.tensors.last().unwrap().name,
        "language_model.model.embed_tokens.weight"
    );
    println!("MINIMAX_RELEASE_PASS parts={PARTS} tensors={TENSORS}");
}

#[test]
#[ignore = "requires the release JSON assets in DS4_MINIMAX_MODEL; no GPU"]
fn source_tokenizer_matches_the_reference_ids() {
    let vocab = Vocab::load_path(&root(), ModelFamily::MinimaxVq).unwrap();
    assert_eq!(vocab.eos_id, 200_020);
    assert!(vocab.is_stop(200_020) && !vocab.is_stop(200_060));
    for case in cases()["cases"].as_array().unwrap() {
        let want: Vec<i32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let got = vocab.encode_rendered_chat(case["text"].as_str().unwrap());
        assert_eq!(got, want, "{}", case["id"]);
    }
}

#[test]
#[ignore = "requires the release JSON assets in DS4_MINIMAX_MODEL; no GPU"]
fn official_template_renders_the_reference_text() {
    let root = root();
    let template = ds4_core::chat_template::Template::from_model(&root)
        .unwrap()
        .unwrap();
    for case in cases()["cases"].as_array().unwrap() {
        if case["context"].is_null() {
            continue;
        }
        let text = template.render(&case["context"]).unwrap();
        assert_eq!(text, case["text"].as_str().unwrap(), "{}", case["id"]);
    }
}
