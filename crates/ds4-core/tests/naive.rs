//! Pinned Naive shape; DSA must retain its own family identity.
use ds4_core::{
    expected_layouts, identify_file, route_architecture, shape_for_variant, validate_file,
    ArchRoute, GgufFile, ModelFamily, TypeClass,
};
use serde_json::{json, Value};
use std::path::PathBuf;

const ARCH: &[u8] = b"naive_n05_flash";

#[test]
fn naive_has_a_distinct_dsa_shape() {
    let ArchRoute::Fixed(variant) = route_architecture(Some(ARCH)) else {
        panic!("Naive architecture is missing from the catalog");
    };
    let shape = shape_for_variant(variant);
    assert_eq!(shape.family.oracle_name().as_bytes(), ARCH);
    assert_eq!(
        ModelFamily::from_oracle_name("naive_n05_flash"),
        Some(shape.family)
    );
    assert_eq!(
        (shape.n_layer, shape.n_embd, shape.n_vocab),
        (48, 4096, 152576)
    );
    assert_eq!((shape.n_head, shape.n_head_kv), (64, 4));
    assert_eq!(
        (shape.n_head_dim, shape.n_value_dim, shape.n_rot),
        (192, 128, 64)
    );
    assert_eq!(
        (
            shape.n_indexer_head,
            shape.n_indexer_head_dim,
            shape.n_indexer_top_k
        ),
        (16, 128, 2048)
    );
    assert_eq!(
        (shape.n_expert, shape.n_expert_used, shape.n_ff_exp),
        (256, 8, 2048)
    );
    assert_eq!(
        (
            shape.n_leading_dense,
            shape.n_expert_shared,
            shape.n_nextn_predict
        ),
        (1, 0, 0)
    );
    assert_eq!(shape.rms_eps, 1e-5);
    assert_eq!(shape.rope_orig_ctx, 1_048_576);
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/naive-main.json")).unwrap()
}

fn put_str(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn put_value(buf: &mut Vec<u8>, typ: u32, value: &Value) {
    match typ {
        2 => buf.extend_from_slice(&(value.as_u64().unwrap() as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&(value.as_u64().unwrap() as u32).to_le_bytes()),
        5 => buf.extend_from_slice(&(value.as_i64().unwrap() as i32).to_le_bytes()),
        6 => buf.extend_from_slice(&(value.as_f64().unwrap() as f32).to_le_bytes()),
        7 => buf.push(u8::from(value.as_bool().unwrap())),
        8 => put_str(buf, value.as_str().unwrap()),
        9 => {
            let item_type = value["item_type"].as_u64().unwrap() as u32;
            let items = value["items"].as_array().unwrap();
            buf.extend_from_slice(&item_type.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                put_value(buf, item_type, item);
            }
        }
        10 => buf.extend_from_slice(&value.as_u64().unwrap().to_le_bytes()),
        12 => buf.extend_from_slice(&value.as_f64().unwrap().to_le_bytes()),
        _ => panic!("unsupported fixture type {typ}"),
    }
}

fn metadata_file(tag: &str, metadata: &Value) -> PathBuf {
    let mut bytes = Vec::from(*b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let entries = metadata.as_object().unwrap();
    bytes.extend_from_slice(&(entries.len() as u64 + 1).to_le_bytes());
    for (key, entry) in entries {
        put_str(&mut bytes, key);
        let typ = entry["type"].as_u64().unwrap() as u32;
        bytes.extend_from_slice(&typ.to_le_bytes());
        put_value(&mut bytes, typ, &entry["value"]);
    }

    // Only cardinality matters here; token strings have separate real goldens.
    put_str(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&152576u64.to_le_bytes());
    bytes.resize(bytes.len() + 152576 * 8, 0);

    let path = std::env::temp_dir().join(format!("ds4-naive-{}-{tag}.gguf", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn check_metadata(tag: &str, metadata: &Value) -> Result<(), String> {
    let path = metadata_file(tag, metadata);
    let file = GgufFile::open(&path).unwrap();
    let result = identify_file(&file)
        .map_err(|error| error.to_string())
        .and_then(|id| validate_file(&file, &id.shape).map_err(|error| error.to_string()));
    drop(file);
    std::fs::remove_file(path).unwrap();
    result
}

#[test]
fn public_metadata_is_accepted() {
    check_metadata("valid", &fixture()["metadata"]).unwrap();
}

#[test]
fn changed_attention_contract_is_rejected() {
    let base = fixture()["metadata"].clone();
    for (index, (key, value)) in [
        ("naive_n05_flash.attention.dsa_enabled", json!(false)),
        ("naive_n05_flash.attention.add_full_sink", json!(true)),
        ("naive_n05_flash.attention.add_swa_sink", json!(false)),
        (
            "naive_n05_flash.attention.layer_norm_rms_epsilon",
            json!(1e-6),
        ),
        ("naive_n05_flash.attention.value_scale", json!(1.0)),
        ("naive_n05_flash.rope.pairing", json!("interleaved")),
        ("naive_n05_flash.index_n_heads", json!(64)),
        ("naive_n05_flash.index_top_k", json!(512)),
        ("naive_n05_flash.indexer_activation_dtype", json!("bf16")),
        ("naive_n05_flash.expert_weights_norm", json!(false)),
        ("general.source.huggingface.revision", json!("unqualified")),
    ]
    .into_iter()
    .enumerate()
    {
        let mut metadata = base.clone();
        metadata[key]["value"] = value;
        let error = check_metadata(&format!("mutation-{index}"), &metadata).unwrap_err();
        assert!(error.contains(key), "{key}: {error}");
    }

    let key = "naive_n05_flash.attention.hybrid_layer_pattern";
    let mut metadata = base;
    metadata[key]["value"]["items"][0] = json!(1);
    assert!(check_metadata("schedule", &metadata)
        .unwrap_err()
        .contains(key));
}

#[test]
fn layouts_match_the_public_directory() {
    let ArchRoute::Fixed(variant) = route_architecture(Some(ARCH)) else {
        panic!("missing family");
    };
    let specs = expected_layouts(&shape_for_variant(variant));
    let source = fixture();
    let tensors = source["tensors"].as_array().unwrap();
    assert_eq!(specs.len(), tensors.len());
    for tensor in tensors {
        let name = tensor["name"].as_str().unwrap();
        let spec = specs
            .iter()
            .find(|spec| spec.name == name)
            .unwrap_or_else(|| panic!("{name}"));
        assert_eq!(
            spec.class,
            TypeClass::Exact(tensor["type"].as_u64().unwrap() as u32),
            "{name}"
        );
        let dims: Vec<_> = tensor["dims"]
            .as_array()
            .unwrap()
            .iter()
            .map(|dim| dim.as_u64().unwrap())
            .collect();
        assert_eq!(spec.ndim as usize, dims.len(), "{name}");
        assert_eq!(spec.dim[..dims.len()], dims, "{name}");
    }
}

#[test]
fn official_template_matches_source() {
    let source =
        include_str!("../../../tests/fixtures/chat-template/models/naive/chat_template.jinja");
    let template = ds4_core::chat_template::Template::compile(
        source,
        ds4_core::chat_template::RenderClock::Fixed(0),
    )
    .unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/naive-tokenizer.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        assert_eq!(
            template.render(&case["context"]).unwrap(),
            case["prompt"].as_str().unwrap(),
            "{}",
            case["id"]
        );
    }
}

#[test]
fn public_tokenizer_matches_source() {
    // This gate reads metadata only. A header-only capture can run it before
    // the 87 GB download completes; it does not qualify model execution.
    let Ok(path) = std::env::var("NAIVE_TOKENIZER_GGUF") else {
        return;
    };
    let file = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let id = identify_file(&file).unwrap();
    let vocab = ds4_core::Vocab::load(&file, id.shape.family).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/naive-tokenizer.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        assert_eq!(
            json!(vocab.encode_rendered_chat(case["prompt"].as_str().unwrap())),
            case["token_ids"],
            "{}",
            case["id"]
        );
    }
    assert!(vocab.is_stop(151645));
    assert!(vocab.is_stop(151643));
    for control in [151644, 151657, 151658, 151667, 151668, 151672] {
        assert!(!vocab.is_stop(control));
    }
    assert!(vocab
        .chat_begin(&mut ds4_core::TokenBuffer::default())
        .is_err());
}
