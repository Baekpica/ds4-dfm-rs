//! Published source-layout manifests are independent of the runtime catalog.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ds4_core::{
    bind_names, catalog_from_bind_name, expected_layouts, expected_mtp_layouts, identify_gguf,
    shape_for_variant, tensor_nbytes, validate_gguf, validate_layouts, validate_mtp_layouts,
    BindPlan, TensorInfo, TensorInventory, TypeClass,
};

const CONFIG: &str = include_str!("../../../tests/fixtures/inkling/config.json");
const PROCESSOR: &str = include_str!("../../../tests/fixtures/inkling/processor_config.json");
const MAIN: &str = include_str!("../../../tests/fixtures/inkling/mq85gb.tsv");
const MTP: &str = include_str!("../../../tests/fixtures/inkling/mtp-bf16.tsv");
static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new(config: &str, layout: &str, sidecar: bool) -> Self {
        let path = std::env::temp_dir().join(format!(
            "ds4-inkling-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut data = b"GGUF".to_vec();
        data.extend(3u32.to_le_bytes());
        data.extend(0u64.to_le_bytes());
        data.extend(9u64.to_le_bytes());
        for (key, val) in [
            ("general.architecture", "inkling"),
            ("inkling.tensor_layout", layout),
            ("inkling.config.json", config),
            ("inkling.processor_config.json", PROCESSOR),
            ("inkling.quantization.recipe", "MQ85GB"),
            (
                "general.source.huggingface.repository",
                "thinkingmachines/Inkling-Small",
            ),
            (
                "general.source.huggingface.revision",
                "8cc5877b44d343f88b92086aa1fb72897950f06a",
            ),
        ] {
            put_str(&mut data, key);
            data.extend(8u32.to_le_bytes());
            put_str(&mut data, val);
        }
        put_str(&mut data, "inkling.mtp.sidecar");
        data.extend(7u32.to_le_bytes());
        data.push(u8::from(sidecar));
        put_str(&mut data, "general.quantization_version");
        data.extend(4u32.to_le_bytes());
        data.extend(2u32.to_le_bytes());
        std::fs::write(&path, data).unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

#[test]
fn identifies_mq85gb() {
    let f = Fixture::new(CONFIG, "source-interleaved-v1", false);
    let id = identify_gguf(&f.0).unwrap();
    assert_eq!(id.shape.family.oracle_name(), "inkling");
    assert_eq!((id.shape.n_layer, id.shape.n_embd), (42, 4096));
    assert_eq!(id.shape.n_vocab, 201024);
    assert_eq!(id.shape.n_expert_shared, 2);
    assert!(!id.shape.use_rope);
    assert_eq!(validate_gguf(&f.0).unwrap(), id.shape);
}

#[test]
fn rejects_attention_schedule() {
    let config = CONFIG.replace("      40\n", "      41\n");
    assert_ne!(config, CONFIG);
    let f = Fixture::new(&config, "source-interleaved-v1", false);
    let error = validate_gguf(&f.0).unwrap_err().to_string();
    assert!(error.contains("local_layer_ids"), "{error}");
}

#[test]
fn rejects_json_and_row_order() {
    for (config, layout, key) in [
        ("{", "source-interleaved-v1", "inkling.config.json"),
        (CONFIG, "gate-then-up", "inkling.tensor_layout"),
    ] {
        let f = Fixture::new(config, layout, false);
        let error = validate_gguf(&f.0).unwrap_err().to_string();
        assert!(error.contains(key), "{error}");
    }
}

#[test]
fn rejects_mtp_as_main_model() {
    let f = Fixture::new(CONFIG, "source-interleaved-v1", true);
    let error = validate_gguf(&f.0).unwrap_err().to_string();
    assert!(error.contains("inkling.mtp.sidecar"), "{error}");
}

fn published(manifest: &str) -> BTreeMap<String, (u32, Vec<u64>)> {
    manifest
        .lines()
        .filter(|s| !s.starts_with('#'))
        .map(|line| {
            let cols: Vec<_> = line.split('\t').collect();
            let dims = cols[2]
                .split(',')
                .map(|n| n.parse().unwrap())
                .rev()
                .collect();
            (cols[0].into(), (cols[1].parse().unwrap(), dims))
        })
        .collect()
}

#[test]
fn matches_published_tensors() {
    for (name, manifest, count) in [("inkling", MAIN, 888), ("mtp-inkling", MTP, 160)] {
        let (support, variant) = catalog_from_bind_name(name).expect("Inkling catalog");
        let shape = shape_for_variant(variant);
        let layouts = if support.is_some() {
            expected_mtp_layouts(&shape)
        } else {
            let names = bind_names(&shape);
            assert_eq!(names.len(), count);
            expected_layouts(&shape)
        };
        let actual: BTreeMap<_, _> = layouts
            .iter()
            .map(|s| {
                let TypeClass::Exact(typ) = s.class else {
                    panic!("artifact precision must be explicit: {}", s.name);
                };
                (s.name.clone(), (typ, s.dim[..s.ndim as usize].to_vec()))
            })
            .collect();
        assert_eq!(layouts.len(), count);
        assert_eq!(actual, published(manifest), "{name}");
    }
}

fn inventory(manifest: &str) -> TensorInventory {
    let tensors = published(manifest)
        .into_iter()
        .map(|(name, (typ, source))| {
            let mut dim = [0; 8];
            dim[..source.len()].copy_from_slice(&source);
            let elements = source.iter().product();
            TensorInfo {
                name,
                ndim: source.len() as u32,
                dim,
                typ,
                elements,
                bytes: tensor_nbytes(typ, elements).unwrap(),
                rel_offset: 0,
                abs_offset: 0,
                shard: 0,
            }
        })
        .collect();
    TensorInventory {
        tensors,
        shards: Vec::new(),
        data_pos: 0,
        alignment: 32,
        page: 4096,
    }
}

#[test]
fn binds_ingress_and_draft() {
    let (_, variant) = catalog_from_bind_name("inkling").unwrap();
    let shape = shape_for_variant(variant);
    let inv = inventory(MAIN);
    assert_eq!(
        inv.tensors.iter().map(|t| t.bytes).sum::<u64>(),
        85_704_616_100
    );
    let mut plan = BindPlan::resolve(shape, &inv);
    assert!(plan.missing_required().is_empty());
    validate_layouts(&plan).unwrap();
    let audio = plan
        .slots
        .iter()
        .position(|s| s.name == "model.audio.encoder.weight")
        .unwrap();
    plan.slots[audio].tensor = None;
    assert!(validate_layouts(&plan)
        .unwrap_err()
        .to_string()
        .contains("model.audio.encoder.weight"));

    let inv = inventory(MTP);
    assert_eq!(
        inv.tensors.iter().map(|t| t.bytes).sum::<u64>(),
        4_463_824_912
    );
    let mut plan = BindPlan::resolve_mtp(shape, &inv);
    assert_eq!(plan.slots.len(), 160);
    assert!(plan.missing_required().is_empty());
    validate_mtp_layouts(&plan).unwrap();
    plan.slots[0].tensor.as_mut().unwrap().typ = 8;
    assert!(validate_mtp_layouts(&plan).is_err(), "MTP must retain BF16");
}

#[test]
fn rejects_wrong_moe_layout() {
    let (_, variant) = catalog_from_bind_name("inkling").unwrap();
    let shape = shape_for_variant(variant);
    let inv = inventory(MAIN);
    let mut plan = BindPlan::resolve(shape, &inv);
    let target = plan
        .slots
        .iter_mut()
        .find(|s| s.name == "model.llm.layers.3.mlp.experts.w2_weight")
        .unwrap();
    target.tensor.as_mut().unwrap().typ = 10;
    assert!(validate_layouts(&plan)
        .unwrap_err()
        .to_string()
        .starts_with("type "));

    let mut plan = BindPlan::resolve(shape, &inv);
    let target = plan
        .slots
        .iter_mut()
        .find(|s| s.name == "model.llm.layers.2.mlp.gate.weight")
        .unwrap();
    target.tensor.as_mut().unwrap().dim[1] = 256;
    assert!(validate_layouts(&plan)
        .unwrap_err()
        .to_string()
        .starts_with("dim "));
}

#[test]
#[ignore = "requires the complete MQ85GB download and MTP-BF16 sidecar"]
fn checks_downloaded_artifacts() {
    let root =
        PathBuf::from(std::env::var("INKLING_ARTIFACT_DIR").expect("set INKLING_ARTIFACT_DIR"));
    let main = root.join("MQ85GB/Inkling-Small-MQ85GB-00001-of-00006.gguf");
    let shape = validate_gguf(&main).unwrap();
    let inv = TensorInventory::open(&main).unwrap();
    assert_eq!(inv.shards.len(), 6);
    assert_eq!(inv.tensors.len(), 888);
    validate_layouts(&BindPlan::resolve(shape, &inv)).unwrap();
    let inv = TensorInventory::open(&root.join("MTP-BF16/Inkling-Small-MTP-BF16.gguf")).unwrap();
    assert_eq!(inv.tensors.len(), 160);
    validate_mtp_layouts(&BindPlan::resolve_mtp(shape, &inv)).unwrap();
}
