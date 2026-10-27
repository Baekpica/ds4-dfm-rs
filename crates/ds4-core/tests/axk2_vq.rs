//! A.X-K2 VQ release contract. Model-backed checks read only JSON assets
//! and GGUF parts (no GPU); point DS4_AXK2_MODEL at the release directory.
use ds4_core::{identify_gguf, ModelFamily, TensorInventory, Variant, Vocab};
use serde_json::Value;
use std::path::PathBuf;

// release.json: 60 expert parts, the body and the embedding part.
const PARTS: usize = 62;
const TENSORS: usize = 47_662;

fn root() -> PathBuf {
    PathBuf::from(std::env::var_os("DS4_AXK2_MODEL").expect("DS4_AXK2_MODEL"))
}

fn cases() -> Value {
    serde_json::from_str(include_str!("../../../tests/fixtures/axk2/chat-cases.json")).unwrap()
}

#[test]
fn family_is_its_own_contract() {
    let shape = ds4_core::shape_for_variant(Variant::Axk2Vq);
    assert_eq!(shape.family, ModelFamily::Axk2Vq);
    assert_eq!(shape.name, "A.X-K2-VQ");
    assert_eq!(
        (shape.n_layer, shape.n_embd, shape.n_vocab),
        (61, 7168, 163_840)
    );
    assert_eq!(
        (shape.n_expert, shape.n_expert_used, shape.n_ff_exp),
        (256, 8, 2048)
    );
    assert_eq!(
        (
            shape.n_indexer_head,
            shape.n_indexer_head_dim,
            shape.n_indexer_top_k
        ),
        (64, 128, 2048)
    );
    let caps = ds4_core::caps_from_shape(shape);
    assert_eq!(caps.ctx_max, Some(262_144));
    assert_eq!(caps.qualified_ctx, None);
}

#[test]
#[ignore = "requires the release in DS4_AXK2_MODEL; hashes every part, no GPU"]
fn release_opens_every_part() {
    let root = root();
    let id = identify_gguf(&root).unwrap();
    assert_eq!(id.shape.family, ModelFamily::Axk2Vq);
    assert_eq!(id.split_count as usize, PARTS);
    let inventory = TensorInventory::open(&root).unwrap();
    assert_eq!(inventory.shards.len(), PARTS);
    assert_eq!(inventory.tensors.len(), TENSORS);
    // The embedding part is the inventory's last; native keeps it on the host.
    let last = inventory.tensors.last().unwrap();
    assert_eq!(last.name, "model.embed_tokens.weight");
    assert_eq!(last.shard as usize, PARTS - 1);
    ds4_core::validate_gguf(&root).unwrap();
    println!("AXK2_RELEASE_PASS parts={PARTS} tensors={TENSORS}");
}

#[test]
#[ignore = "requires the release JSON assets in DS4_AXK2_MODEL; no GPU"]
fn source_tokenizer_matches_the_reference_ids() {
    let vocab = Vocab::load_path(&root(), ModelFamily::Axk2Vq).unwrap();
    // <|im_end|> ends a reply; <|endoftext|> and <|im_start|> never leak.
    assert_eq!(vocab.eos_id, 163_704);
    assert!(vocab.is_stop(163_704) && vocab.is_stop(163_691) && vocab.is_stop(163_702));
    assert!(!vocab.is_stop(163_706));
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
#[ignore = "requires the release JSON assets in DS4_AXK2_MODEL; no GPU"]
fn official_template_renders_the_reference_text() {
    let template = ds4_core::chat_template::Template::from_model(&root())
        .unwrap()
        .unwrap();
    for case in cases()["cases"].as_array().unwrap() {
        let text = template.render(&case["context"]).unwrap();
        assert_eq!(text, case["text"].as_str().unwrap(), "{}", case["id"]);
    }
}
