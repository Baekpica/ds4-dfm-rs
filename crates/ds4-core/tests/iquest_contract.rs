//! Independent contract from the published IQuest metadata and tensor manifest.
use ds4_core::{
    expected_layouts, identify_file, route_architecture, shape_for_variant, validate_file,
    validate_layouts, ArchRoute, BindPlan, GgufFile, ModelFamily, TensorInfo, TensorInventory,
    TypeClass,
};
use serde_json::{json, Value};
use std::path::PathBuf;

const ARCH: &[u8] = b"iquest_q1";
const VOCAB: usize = 160_000;

#[test]
fn recursive_mtp_shape() {
    let ArchRoute::Fixed(variant) = route_architecture(Some(ARCH)) else {
        panic!("IQuest architecture is missing from the catalog");
    };
    let shape = shape_for_variant(variant);
    assert_eq!(shape.family.oracle_name().as_bytes(), ARCH);
    assert_eq!(
        ModelFamily::from_oracle_name("iquest_q1"),
        Some(shape.family)
    );
    assert_eq!(
        (shape.n_layer, shape.n_embd, shape.n_vocab),
        (88, 3072, VOCAB as u32)
    );
    assert_eq!((shape.n_head, shape.n_head_kv), (48, 8));
    assert_eq!(
        (shape.n_head_dim, shape.n_value_dim, shape.n_rot),
        (128, 128, 32)
    );
    assert_eq!(
        (shape.n_expert, shape.n_expert_used, shape.n_ff_exp),
        (256, 8, 1536)
    );
    assert_eq!(
        (
            shape.n_leading_dense,
            shape.n_ff_dense,
            shape.n_expert_shared
        ),
        (1, 12288, 0)
    );
    assert_eq!((shape.n_swa, shape.n_full_attn_count), (4096, 25));
    assert_eq!(shape.n_nextn_predict, 1);
    assert!(shape.use_rope && shape.use_qk_norm);
    assert_eq!(shape.rms_eps, 1e-6);
    assert_eq!(shape.rope_freq_base, 1_000_000.0);
    assert_eq!(shape.rope_freq_base_swa, 10_000.0);
    assert_eq!(shape.rope_orig_ctx, 524_288);
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/iquest-main.json")).unwrap()
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

    // Only vocabulary cardinality belongs in this metadata gate.
    put_str(&mut bytes, "tokenizer.ggml.tokens");
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&(VOCAB as u64).to_le_bytes());
    bytes.resize(bytes.len() + VOCAB * 8, 0);

    let path = std::env::temp_dir().join(format!("ds4-iquest-{}-{tag}.gguf", std::process::id()));
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
fn rejects_graph_changes() {
    let base = fixture()["metadata"].clone();
    check_metadata("mutation-base", &base).unwrap();
    for (index, (key, value)) in [
        ("iquest_q1.rope.pairing", json!("interleaved")),
        ("iquest_q1.rope.dimension_count", json!(128)),
        ("iquest_q1.rope.freq_base_swa", json!(1_000_000.0)),
        ("iquest_q1.attention.sink_type", json!("constant-per-head")),
        ("iquest_q1.attention.sliding_window", json!(512)),
        ("iquest_q1.expert_gating_func", json!(2)),
        ("iquest_q1.expert_weights_norm", json!(false)),
        ("iquest_q1.feed_forward.output_scale", json!(1.0)),
        ("iquest_q1.nextn_predict_layers", json!(7)),
        ("iquest_q1.mtp.shared_embedding", json!("output.weight")),
        (
            "iquest_q1.mtp.shared_target_norm",
            json!("mtp.0.output_norm.weight"),
        ),
        ("iquest_q1.mtp.sliding_window", json!(4096)),
        ("iquest_q1.mtp.draft_slots", json!(8)),
        ("iquest_q1.mtp.fp32_residual", json!(false)),
        ("general.source.huggingface.revision", json!("unqualified")),
        (
            "general.source.huggingface.repository",
            json!("another/model"),
        ),
        ("iquest_q1.tensor_layout", json!("llama-compatible")),
        ("iquest_q1.quantization.minimum", json!("IQ1_S")),
    ]
    .into_iter()
    .enumerate()
    {
        let mut metadata = base.clone();
        metadata[key]["value"] = value;
        assert!(
            check_metadata(&format!("mutation-{index}"), &metadata).is_err(),
            "accepted {key}"
        );
    }

    let mut metadata = base;
    metadata["iquest_q1.attention.full_attention_layers"]["value"]["items"][1] = json!(2);
    assert!(check_metadata("schedule", &metadata).is_err());
}

#[test]
fn layouts_match_directory() {
    let ArchRoute::Fixed(variant) = route_architecture(Some(ARCH)) else {
        panic!("missing family");
    };
    let specs = expected_layouts(&shape_for_variant(variant));
    let source = fixture();
    let tensors = source["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 1254);
    assert_eq!(specs.len(), tensors.len());
    let mut inventory = TensorInventory {
        shards: Vec::new(),
        tensors: Vec::new(),
        data_pos: 0,
        alignment: 32,
        page: 4096,
    };
    for tensor in tensors {
        let name = tensor["name"].as_str().unwrap();
        let spec = specs
            .iter()
            .find(|spec| spec.name == name)
            .unwrap_or_else(|| panic!("{name}"));
        let dims: Vec<_> = tensor["dims"]
            .as_array()
            .unwrap()
            .iter()
            .map(|dim| dim.as_u64().unwrap())
            .collect();
        assert_eq!(spec.ndim as usize, dims.len(), "{name}");
        assert_eq!(spec.dim[..dims.len()], dims, "{name}");
        assert_eq!(
            spec.class,
            TypeClass::Exact(tensor["type"].as_u64().unwrap() as u32),
            "unqualified precision accepted for {name}"
        );
        let mut dim = [0; 8];
        dim[..dims.len()].copy_from_slice(&dims);
        let info = TensorInfo {
            name: name.into(),
            ndim: dims.len() as u32,
            dim,
            typ: tensor["type"].as_u64().unwrap() as u32,
            rel_offset: 0,
            abs_offset: 0,
            elements: dims.iter().product(),
            bytes: 0,
            shard: 0,
        };
        inventory.tensors.push(info);
    }
    let plan = BindPlan::resolve(shape_for_variant(variant), &inventory);
    plan.check().unwrap();
    validate_layouts(&plan).unwrap();
}
