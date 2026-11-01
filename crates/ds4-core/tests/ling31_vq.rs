//! Ling-3.1-flash VQ release contract. Model-backed checks read only JSON
//! assets and GGUF headers (no GPU); point DS4_LING31_MODEL at the release.
use ds4_core::{
    chat_template::Template, identify_gguf, ModelFamily, TensorInventory, Variant, Vocab,
};
use serde_json::{json, Value};
use std::path::PathBuf;

const TRUNK_PARTS: usize = 69;
const TRUNK_TENSORS: usize = 106_041;
const MTP_TENSORS: usize = 107_606;

fn root() -> PathBuf {
    PathBuf::from(std::env::var_os("DS4_LING31_MODEL").expect("DS4_LING31_MODEL"))
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/ling31/evaluation-token-ids.json"
    ))
    .unwrap()
}

fn ids(v: &Value) -> Vec<i32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect()
}

#[test]
fn family_is_its_own_contract() {
    let shape = ds4_core::shape_for_variant(Variant::Ling31FlashVq);
    assert_eq!(shape.family, ModelFamily::Ling31Vq);
    assert_ne!(shape.family, ModelFamily::Ling3Vl);
    assert_eq!(shape.name, "Ling-3.1-flash-VQ");
    assert_eq!(
        (shape.n_layer, shape.n_embd, shape.n_vocab),
        (73, 4096, 157_184)
    );
    assert_eq!(
        (shape.n_expert, shape.n_ff_exp, shape.n_leading_dense),
        (512, 1280, 4)
    );
    let caps = ds4_core::caps_from_shape(shape);
    assert_eq!(caps.ctx_max, Some(1_048_576));
    assert_eq!(ds4_core::native_ctx_default(caps), Some(262_144));
    assert_eq!(caps.qualified_ctx, None);
}

#[test]
#[ignore = "requires the release in DS4_LING31_MODEL; hashes every part, no GPU"]
fn release_features_select_parts() {
    let root = root();
    let id = identify_gguf(&root).unwrap();
    assert_eq!(id.shape.family, ModelFamily::Ling31Vq);
    assert_eq!(id.split_count as usize, TRUNK_PARTS);
    let trunk = TensorInventory::open(&root).unwrap();
    assert_eq!(trunk.shards.len(), TRUNK_PARTS);
    assert_eq!(trunk.tensors.len(), TRUNK_TENSORS);
    assert!(trunk
        .tensors
        .iter()
        .all(|t| !t.name.starts_with("model.layers.72.")));
    ds4_core::validate_gguf(&root).unwrap();
    let mtp = TensorInventory::open_ling31(&root, ds4_core::Ling31Feature::Mtp).unwrap();
    assert_eq!(mtp.shards.len(), TRUNK_PARTS + 2);
    assert_eq!(mtp.tensors.len(), MTP_TENSORS);
    println!("LING31_RELEASE_PASS trunk={TRUNK_TENSORS} mtp={MTP_TENSORS}");
}

#[test]
#[ignore = "requires the 1019 release in DS4_LING31_MODEL; hashes every part, no GPU"]
fn release_1019_all12_parts() {
    let root = root();
    let id = identify_gguf(&root).unwrap();
    assert_eq!(id.split_count, 70);
    ds4_core::validate_gguf(&root).unwrap();
    let trunk = TensorInventory::open_ling31(&root, ds4_core::Ling31Feature::Trunk).unwrap();
    assert_eq!(trunk.shards.len(), 70);
    assert!(trunk
        .tensors
        .iter()
        .all(|t| !t.name.starts_with("model.layers.72.")));
    let mtp = TensorInventory::open_ling31(&root, ds4_core::Ling31Feature::Mtp).unwrap();
    assert_eq!(mtp.shards.len(), 72);
    assert_eq!(mtp.tensors.len(), 107_480);
    assert_eq!(
        mtp.tensors.last().unwrap().name,
        "model.word_embeddings.weight"
    );
    let books = mtp
        .tensors
        .iter()
        .filter(|t| t.name.ends_with(".mlp.vq_codebook"))
        .collect::<Vec<_>>();
    assert_eq!(books.len(), 69);
    assert!(books.iter().all(|t| t.dim[0] == 8 && t.dim[1] == 4096));
    println!("LING31_1019_ALL12_PASS parts=72 tensors=107480 books=69");
}

#[test]
#[ignore = "requires the release JSON assets in DS4_LING31_MODEL; no GPU"]
fn official_template_and_tokenizer() {
    let root = root();
    let fixture = fixture();
    let vocab = Vocab::load_path(&root, ModelFamily::Ling31Vq).unwrap();
    let template = Template::from_model(&root).unwrap().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let mut context = json!({
            "messages": case["messages"],
            "add_generation_prompt": true,
            "enable_thinking": case["enable_thinking"],
        });
        if let Some(tools) = case.get("tools") {
            context["tools"] = tools.clone();
        }
        let text = template.render(&context).unwrap();
        assert_eq!(
            text,
            case["rendered_prompt"].as_str().unwrap(),
            "{}",
            case["id"]
        );
        assert_eq!(
            vocab.encode_rendered_chat(&text),
            ids(&case["prompt_ids"]),
            "{}",
            case["id"]
        );
        let target = case["target"].as_str().unwrap();
        assert_eq!(
            vocab.encode_rendered_chat(target),
            ids(&case["target_ids"]),
            "{}",
            case["id"]
        );
    }
}

#[test]
#[ignore = "requires the release in DS4_LING31_MODEL; hashes every part twice, no GPU"]
fn check_config_probes_requested_parts() {
    // A copy of the release without its MTP parts still serves the trunk;
    // a check for MTP serving must refuse it before listen.
    let root = root();
    let copy = std::env::temp_dir().join(format!("ling31-probe-{}", std::process::id()));
    std::fs::create_dir_all(&copy).unwrap();
    for entry in std::fs::read_dir(&root).unwrap() {
        let name = entry.unwrap().file_name();
        if name != "mtp-native.gguf" && name != "layer-72-experts.gguf" {
            std::os::unix::fs::symlink(root.join(&name), copy.join(&name)).unwrap();
        }
    }
    let path = copy.to_str().unwrap();
    let trunk = ds4_core::probe_model_artifact(path, ds4_core::ProbeParts::Trunk);
    let mtp = ds4_core::probe_model_artifact(path, ds4_core::ProbeParts::WithMtp);
    std::fs::remove_dir_all(&copy).unwrap();
    assert!(trunk.is_ok(), "{:?}", trunk.err());
    assert!(mtp.is_err());
}
