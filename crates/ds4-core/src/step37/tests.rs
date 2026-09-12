use super::*;
use crate::tensors::{tensor_nbytes, ShardPlan};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

const METADATA: &str = include_str!("../../../../tests/fixtures/step37/metadata.json");
const INVENTORY: &str = include_str!("../../../../tests/fixtures/step37/mq83.tsv");
static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Fixture(PathBuf);

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

fn value(out: &mut Vec<u8>, typ: &str, v: &Value) {
    match typ {
        "UINT16" => out.extend((v.as_u64().unwrap() as u16).to_le_bytes()),
        "UINT32" => out.extend((v.as_u64().unwrap() as u32).to_le_bytes()),
        "INT32" => out.extend((v.as_i64().unwrap() as i32).to_le_bytes()),
        "FLOAT32" => out.extend((v.as_f64().unwrap() as f32).to_le_bytes()),
        "BOOL" => out.push(u8::from(v.as_bool().unwrap())),
        "STRING" => string(out, v.as_str().unwrap()),
        _ => panic!("unexpected fixture type {typ}"),
    }
}

fn type_id(typ: &str) -> u32 {
    match typ {
        "UINT16" => 2,
        "UINT32" => 4,
        "INT32" => 5,
        "FLOAT32" => 6,
        "BOOL" => 7,
        "STRING" => 8,
        "ARRAY" => 9,
        _ => panic!("unexpected fixture type {typ}"),
    }
}

impl Fixture {
    fn new(change: impl FnOnce(&mut Value)) -> Self {
        Self::from_metadata(METADATA, change)
    }

    fn from_metadata(source: &str, change: impl FnOnce(&mut Value)) -> Self {
        let mut metadata: Value = serde_json::from_str(source).unwrap();
        change(&mut metadata);
        let metadata = metadata.as_array().unwrap();
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend((metadata.len() as u64 + 1).to_le_bytes());
        for row in metadata {
            string(&mut out, row[0].as_str().unwrap());
            let typ = row[1][0].as_str().unwrap();
            out.extend(type_id(typ).to_le_bytes());
            if typ != "ARRAY" {
                value(&mut out, typ, &row[2]);
                continue;
            }
            let subtype = row[1][1].as_str().unwrap();
            out.extend(type_id(subtype).to_le_bytes());
            let values = row[2].as_array().unwrap();
            out.extend((values.len() as u64).to_le_bytes());
            for v in values {
                value(&mut out, subtype, v);
            }
        }
        // Token contents belong to tokenizer tests; preflight checks the count.
        string(&mut out, "tokenizer.ggml.tokens");
        out.extend(9u32.to_le_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend(VOCAB.to_le_bytes());
        out.resize(out.len() + VOCAB as usize * 8, 0);
        let path = std::env::temp_dir().join(format!(
            "ds4-step37-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, out).unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn inventory() -> TensorInventory {
    fixture_inventory(INVENTORY)
}

fn fixture_inventory(source: &str) -> TensorInventory {
    let mut offset = 4096;
    let tensors = source
        .lines()
        .map(|line| {
            let parts: Vec<_> = line.split('\t').collect();
            let typ = parts[1].parse().unwrap();
            let dims: Vec<u64> = parts[2].split(',').map(|s| s.parse().unwrap()).collect();
            let elements = dims.iter().product();
            let bytes = tensor_nbytes(typ, elements).unwrap();
            let mut dim = [0; 8];
            dim[..dims.len()].copy_from_slice(&dims);
            let t = TensorInfo {
                name: parts[0].into(),
                ndim: dims.len() as u32,
                dim,
                typ,
                rel_offset: offset - 4096,
                abs_offset: offset,
                elements,
                bytes,
                shard: 0,
            };
            offset += bytes;
            t
        })
        .collect();
    TensorInventory {
        shards: vec![ShardPlan {
            path: PathBuf::new(),
            size: offset,
            base: 0,
        }],
        tensors,
        data_pos: 4096,
        alignment: 32,
        page: 4096,
    }
}

#[test]
fn metadata_and_layer_schedule() {
    let fixture = Fixture::new(|_| {});
    check_metadata(&GgufFile::open(&fixture.0).unwrap()).unwrap();
    assert_eq!(Step37Layer { index: 0 }.query_heads(), 64);
    assert_eq!(Step37Layer { index: 1 }.query_heads(), 96);
    assert_eq!(Step37Layer { index: 44 }.rotary_dims(), 64);
    assert_eq!(Step37Layer { index: 43 }.rotary_dims(), 128);
    assert_eq!(Step37Layer { index: 42 }.swiglu_clamps(), (0.0, 0.0));
    assert_eq!(Step37Layer { index: 43 }.swiglu_clamps(), (7.0, 16.0));
    assert_eq!(Step37Layer { index: 44 }.sliding_window(), None);
    assert_eq!(Step37Layer { index: 43 }.sliding_window(), Some(512));
    // The architecture has its own family; it cannot fall through to DeepSeek.
    assert_eq!(
        crate::shape::route_architecture(Some(b"step35")),
        crate::shape::ArchRoute::Fixed(crate::shape::Variant::Step37Flash)
    );
}

#[test]
fn rejects_wrong_mechanics() {
    for (key, replacement) in [
        ("step35.block_count", serde_json::json!(48)),
        ("step35.expert_gating_func", serde_json::json!(1)),
        ("step35.expert_weights_norm", serde_json::json!(false)),
        ("step35.rope.freq_base_swa", serde_json::json!(5000000.0)),
        ("step37.source_revision", serde_json::json!("different")),
        (
            "step35.attention.head_count",
            serde_json::to_value(vec![64; 45]).unwrap(),
        ),
    ] {
        let fixture = Fixture::new(|rows| {
            let row = rows
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|r| r[0] == key)
                .unwrap();
            row[2] = replacement;
        });
        let err = check_metadata(&GgufFile::open(&fixture.0).unwrap()).unwrap_err();
        assert!(err.to_string().contains(key), "{err}");
    }
}

#[test]
fn binds_mq83_fixture() {
    let mut inv = inventory();
    // Binding is by name, not source order or shard enumeration order.
    inv.tensors.reverse();
    let indices = check_tensors(&inv).unwrap();
    assert_eq!(indices.len(), 754);
    assert_eq!(
        indices.iter().map(|&i| inv.tensors[i].bytes).sum::<u64>(),
        83_001_512_448
    );
}

#[test]
fn rejects_tensor_contracts() {
    for name in [
        "blk.7.ffn_gate_exps.weight",
        "blk.40.ffn_up_exps.weight",
        "blk.41.ffn_gate_exps.weight",
        "blk.3.ffn_down_shexp.weight",
        "blk.44.attn_gate.weight",
    ] {
        let mut inv = inventory();
        let i = inv.find_index(name).unwrap();
        inv.tensors[i].typ = F32;
        assert!(check_tensors(&inv).unwrap_err().to_string().contains(name));
    }
    let mut inv = inventory();
    let i = inv.find_index("blk.1.attn_q.weight").unwrap();
    inv.tensors[i].dim[1] = 8192;
    assert!(check_tensors(&inv)
        .unwrap_err()
        .to_string()
        .contains("blk.1.attn_q.weight"));
    let mut inv = inventory();
    inv.tensors[1].name = inv.tensors[0].name.clone();
    assert!(check_tensors(&inv)
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
    let mut inv = inventory();
    inv.tensors[1].abs_offset = inv.tensors[0].abs_offset;
    assert!(check_tensors(&inv)
        .unwrap_err()
        .to_string()
        .contains("overlapping"));
    inv.tensors.pop();
    assert!(check_tensors(&inv)
        .unwrap_err()
        .to_string()
        .contains("tensor count"));
}

const MTP_META: &str = include_str!("../../../../tests/fixtures/step37/mtp-metadata.json");
const MTP_TENSORS: &str = include_str!("../../../../tests/fixtures/step37/mtp.tsv");
const VISION_META: &str = include_str!("../../../../tests/fixtures/step37/vision-metadata.json");
const VISION_TENSORS: &str = include_str!("../../../../tests/fixtures/step37/vision.tsv");

#[test]
fn sidecar_metadata_and_bindings() {
    for (kind, metadata, tensors, count) in [
        (Step37Sidecar::Mtp, MTP_META, MTP_TENSORS, 55),
        (Step37Sidecar::Vision, VISION_META, VISION_TENSORS, 667),
    ] {
        let f = Fixture::from_metadata(metadata, |_| {});
        check_sidecar(&GgufFile::open(&f.0).unwrap(), kind).unwrap();
        let mut inv = fixture_inventory(tensors);
        inv.tensors.reverse();
        let bound = check_specs(&inv, sidecar_specs(kind)).unwrap();
        assert_eq!(bound.len(), count);
        let unique: BTreeSet<_> = bound.into_iter().collect();
        assert_eq!(unique.len(), count);
    }
}

#[test]
fn rejects_wrong_sidecar_semantics() {
    for (kind, metadata, key, value) in [
        (
            Step37Sidecar::Mtp,
            MTP_META,
            "step35.nextn_predict_layers",
            serde_json::json!(1),
        ),
        (
            Step37Sidecar::Mtp,
            MTP_META,
            "step35.block_count",
            serde_json::json!(45),
        ),
        (
            Step37Sidecar::Mtp,
            MTP_META,
            "step35.attention.head_count",
            serde_json::json!(vec![64; 48]),
        ),
        (
            Step37Sidecar::Vision,
            VISION_META,
            "clip.projector_type",
            serde_json::json!("qwen2vl"),
        ),
        (
            Step37Sidecar::Vision,
            VISION_META,
            "clip.vision.projector.scale_factor",
            serde_json::json!(2),
        ),
        (
            Step37Sidecar::Vision,
            VISION_META,
            "clip.vision.image_mean",
            serde_json::json!([0.5, 0.5, 0.5]),
        ),
    ] {
        let f = Fixture::from_metadata(metadata, |rows| {
            rows.as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|r| r[0] == key)
                .unwrap()[2] = value;
        });
        let err = check_sidecar(&GgufFile::open(&f.0).unwrap(), kind).unwrap_err();
        assert!(err.to_string().contains(key), "{err}");
    }
}

#[test]
fn mtp_heads_must_remain_independent() {
    let mut inv = fixture_inventory(MTP_TENSORS);
    let first = inv
        .find_index("blk.45.nextn.shared_head_head.weight")
        .unwrap();
    let second = inv
        .find_index("blk.46.nextn.shared_head_head.weight")
        .unwrap();
    inv.tensors[second].abs_offset = inv.tensors[first].abs_offset;
    assert!(check_specs(&inv, sidecar_specs(Step37Sidecar::Mtp))
        .unwrap_err()
        .to_string()
        .contains("overlapping"));
    let mut inv = fixture_inventory(VISION_TENSORS);
    let i = inv.find_index("mm.0.weight").unwrap();
    inv.tensors[i].dim[0] = 1;
    assert!(check_specs(&inv, sidecar_specs(Step37Sidecar::Vision))
        .unwrap_err()
        .to_string()
        .contains("mm.0.weight"));
}

#[test]
fn host_catalog_uses_exact_contract() {
    let f = Fixture::new(|_| {});
    let g = GgufFile::open(&f.0).unwrap();
    let id = crate::identify::identify_file(&g).unwrap();
    assert_eq!(id.shape.family, crate::shape::ModelFamily::Step37);
    assert_eq!((id.shape.n_head, id.shape.n_swa_head), (64, 96));
    crate::validate::validate_file(&g, &id.shape).unwrap();
    let inv = inventory();
    let plan = crate::bind::BindPlan::resolve(id.shape, &inv);
    assert_eq!(plan.slots.len(), 754);
    assert!(plan.missing_required().is_empty());
    crate::layout::validate_layouts(&plan).unwrap();
}
