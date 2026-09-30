//! External DSpark has its own graph and must pair with the pinned target.
use ds4_core::{probe_dspark_sidecar, shape_for_variant, tensor_nbytes, Variant};
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/naive-draft.json")).unwrap()
}

fn put_str(buf: &mut Vec<u8>, text: &str) {
    buf.extend_from_slice(&(text.len() as u64).to_le_bytes());
    buf.extend_from_slice(text.as_bytes());
}

fn put_value(buf: &mut Vec<u8>, typ: u32, value: &Value) {
    match typ {
        4 => buf.extend_from_slice(&(value.as_u64().unwrap() as u32).to_le_bytes()),
        5 => buf.extend_from_slice(&(value.as_i64().unwrap() as i32).to_le_bytes()),
        6 => buf.extend_from_slice(&(value.as_f64().unwrap() as f32).to_le_bytes()),
        7 => buf.push(u8::from(value.as_bool().unwrap())),
        8 => put_str(buf, value.as_str().unwrap()),
        9 => {
            let typ = value["item_type"].as_u64().unwrap() as u32;
            let items = value["items"].as_array().unwrap();
            buf.extend_from_slice(&typ.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                put_value(buf, typ, item);
            }
        }
        _ => panic!("unsupported fixture type {typ}"),
    }
}

fn sparse_file(tag: &str, source: &Value) -> PathBuf {
    let metadata = source["metadata"].as_object().unwrap();
    let tensors = source["tensors"].as_array().unwrap();
    let mut buf = Vec::from(*b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, entry) in metadata {
        put_str(&mut buf, key);
        let typ = entry["type"].as_u64().unwrap() as u32;
        buf.extend_from_slice(&typ.to_le_bytes());
        put_value(&mut buf, typ, &entry["value"]);
    }
    let mut end = 0;
    for tensor in tensors {
        put_str(&mut buf, tensor["name"].as_str().unwrap());
        let dims = tensor["dims"].as_array().unwrap();
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        let mut elements = 1;
        for dim in dims {
            let dim = dim.as_u64().unwrap();
            buf.extend_from_slice(&dim.to_le_bytes());
            elements *= dim;
        }
        let typ = tensor["type"].as_u64().unwrap() as u32;
        let offset = tensor["offset"].as_u64().unwrap();
        buf.extend_from_slice(&typ.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        end = end.max(offset + tensor_nbytes(typ, elements).unwrap());
    }
    buf.resize(buf.len().div_ceil(32) * 32, 0);
    let path =
        std::env::temp_dir().join(format!("ds4-naive-draft-{}-{tag}.gguf", std::process::id()));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&buf).unwrap();
    file.set_len(buf.len() as u64 + end).unwrap();
    path
}

fn probe(tag: &str, source: &Value) -> Result<(), String> {
    let path = sparse_file(tag, source);
    let shape = shape_for_variant(Variant::NaiveN05Flash);
    let result = probe_dspark_sidecar(shape, None, path.to_str().unwrap())
        .map_err(|error| error.to_string());
    std::fs::remove_file(path).unwrap();
    result
}

#[test]
fn pinned_dspark_is_accepted() {
    probe("valid", &fixture()).unwrap();
}

#[test]
fn draft_inspect_uses_own_graph() {
    use ds4_core::{dump_bind_names_variant, dump_expected_layouts_variant};
    let names = dump_bind_names_variant("dspark-naive-n05-flash").unwrap();
    let layout = dump_expected_layouts_variant("dspark-naive-n05-flash").unwrap();
    assert!(names.contains("n_layer=5"));
    assert!(names.contains("fc.weight"));
    assert!(!names.contains("dspark.main_proj.weight"));
    assert!(layout.contains("markov_rank=256 n_layer=5"));
}

#[test]
fn changed_draft_contract_fails() {
    let base = fixture();
    for (i, (key, value)) in [
        ("general.architecture", json!("dflash")),
        ("naive_draft.target.revision", json!("other")),
        ("naive_draft.source.revision", json!("other")),
        ("naive_draft.target.hidden_state_offset", json!(0)),
        ("naive_draft.block_size", json!(8)),
        ("naive_draft.proposal_count", json!(7)),
        ("naive_draft.attention.head_count", json!(16)),
        ("naive_draft.attention.sliding_window", json!(128)),
        ("naive_draft.attention.causal", json!(true)),
        ("naive_draft.use_target_kv", json!(true)),
        ("naive_draft.use_mask_embedding", json!(false)),
        ("naive_draft.markov_head_type", json!("rnn")),
        ("naive_draft.confidence_head_with_markov", json!(false)),
    ]
    .into_iter()
    .enumerate()
    {
        let mut source = base.clone();
        source["metadata"][key]["value"] = value;
        assert!(probe(&format!("metadata-{i}"), &source)
            .unwrap_err()
            .contains(key));
    }
    let mut source = base;
    source["metadata"]["naive_draft.target.layer_ids"]["value"]["items"][0] = json!(2);
    assert!(probe("taps", &source).unwrap_err().contains("layer_ids"));
}

#[test]
fn changed_draft_layout_fails() {
    let base = fixture();
    for (i, tensor) in base["tensors"].as_array().unwrap().iter().enumerate() {
        let mut source = base.clone();
        source["tensors"][i]["dims"][0] = json!(32);
        assert!(probe(&format!("layout-{i}"), &source)
            .unwrap_err()
            .contains(tensor["name"].as_str().unwrap()));
    }
    let mut source = base;
    source["tensors"].as_array_mut().unwrap().pop();
    assert!(probe("missing", &source).is_err());
}

#[test]
fn downloaded_draft_probe() {
    let Ok(path) = std::env::var("NAIVE_DRAFT_GGUF") else {
        return;
    };
    probe_dspark_sidecar(shape_for_variant(Variant::NaiveN05Flash), None, &path).unwrap();
}

#[test]
fn native_draft_matches_directory() {
    use std::process::{Command, Stdio};

    let Ok(path) = std::env::var("NAIVE_DRAFT_BIND") else {
        return;
    };
    let mut child = Command::new(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    for tensor in fixture()["tensors"].as_array().unwrap() {
        let dims = tensor["dims"].as_array().unwrap();
        writeln!(
            input,
            "{} {} {} {} {}",
            tensor["name"].as_str().unwrap(),
            tensor["type"],
            dims.len(),
            dims[0],
            dims.get(1).unwrap_or(&json!(0))
        )
        .unwrap();
    }
    drop(input);
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap().trim(),
        "63 native draft bindings; source dimensions and formats"
    );
}
