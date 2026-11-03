//! Gemma-4 26B A4B VQ release contract. Model-backed checks read only JSON
//! assets and GGUF parts (no GPU); point DS4_GEMMA4_MODEL at the release
//! directory.
use ds4_core::{identify_gguf, ModelFamily, TensorInventory, Variant, Vocab};
use serde_json::Value;
use std::path::PathBuf;

// release.json: the trunk and the embedding part; the drafter is a third,
// optional part.
const PARTS: usize = 2;
const TENSORS: usize = 12_177;

fn root() -> PathBuf {
    PathBuf::from(std::env::var_os("DS4_GEMMA4_MODEL").expect("DS4_GEMMA4_MODEL"))
}

fn cases() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/gemma4/eval-cases.json"
    ))
    .unwrap()
}

#[test]
fn family_is_its_own_contract() {
    let shape = ds4_core::shape_for_variant(Variant::Gemma4A4bVq);
    assert_eq!(shape.family, ModelFamily::Gemma4Vq);
    assert_eq!(shape.name, "Gemma-4-26B-A4B-VQ");
    assert_eq!(
        (shape.n_layer, shape.n_embd, shape.n_vocab),
        (30, 2816, 262_144)
    );
    assert_eq!(
        (shape.n_head, shape.n_head_kv, shape.n_head_dim, shape.n_rot),
        (16, 8, 256, 64)
    );
    assert_eq!(
        (
            shape.n_expert,
            shape.n_expert_used,
            shape.n_ff_exp,
            shape.n_ff_dense
        ),
        (128, 8, 704, 2112)
    );
    // The artifact states the model's own maximum; nothing is qualified
    // until a live gate measures it on this host.
    let caps = ds4_core::caps_from_shape(shape);
    assert_eq!(caps.ctx_max, Some(262_144));
    assert_eq!(caps.qualified_ctx, None);
}

#[test]
#[ignore = "requires the release in DS4_GEMMA4_MODEL; hashes every part, no GPU"]
fn release_opens_every_part() {
    let root = root();
    let id = identify_gguf(&root).unwrap();
    assert_eq!(id.shape.family, ModelFamily::Gemma4Vq);
    assert_eq!(id.split_count as usize, PARTS);
    let inventory = TensorInventory::open(&root).unwrap();
    assert_eq!(inventory.shards.len(), PARTS);
    assert_eq!(inventory.tensors.len(), TENSORS);
    // The embedding part is the inventory's last; native keeps it on the host.
    let last = inventory.tensors.last().unwrap();
    assert_eq!(last.name, "model.embed_tokens.weight");
    assert_eq!(last.shard as usize, PARTS - 1);
    ds4_core::validate_gguf(&root).unwrap();
    println!("GEMMA4_RELEASE_PASS parts={PARTS} tensors={TENSORS}");
}

#[test]
#[ignore = "requires the release in DS4_GEMMA4_MODEL; no GPU"]
fn source_tokenizer_matches_the_reference_ids() {
    let vocab = Vocab::load_path(&root(), ModelFamily::Gemma4Vq).unwrap();
    assert_eq!((vocab.bos_id, vocab.eos_id), (2, 1));
    assert!(vocab.is_stop(1) && vocab.is_stop(106) && vocab.is_stop(50));
    assert!(!vocab.is_stop(105));
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
#[ignore = "requires the release in DS4_GEMMA4_MODEL; no GPU"]
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

/// The artifact's trunk carries the official template verbatim under GGUF's
/// `tokenizer.chat_template`, and the engine must render it to the same bytes
/// the reference engine produced from the checkpoint's `chat_template.jinja`.
#[test]
#[ignore = "requires the release in DS4_GEMMA4_MODEL; reads metadata only"]
fn artifact_carries_the_official_chat_template() {
    let root = root();
    let template = ds4_core::chat_template::Template::from_model(&root)
        .unwrap()
        .expect("the artifact must carry a chat template");
    assert!(
        template.source().contains("tokenizer.chat_template"),
        "template came from {}",
        template.source()
    );
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/gemma4/server-vectors.json"
    ))
    .unwrap();
    for case in fixtures["vectors"].as_array().unwrap() {
        let Some(expected) = case["expected"].as_str() else {
            continue;
        };
        assert_eq!(
            template.render(&case["context"]).unwrap(),
            expected,
            "{}",
            case["name"]
        );
    }
    // generation_config.json eos_token_id is [1, 106, 50].
    let vocab = Vocab::load_path(&root, ModelFamily::Gemma4Vq).unwrap();
    assert!(vocab.is_stop(1) && vocab.is_stop(106) && vocab.is_stop(50));
    assert!(!vocab.is_stop(200));
}
