//! V4.1 wire and bind gates, model-free.
//!
//! The metadata block is synthetic but carries the artifact's own arrays,
//! read from `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf` over the Spark
//! mount on 2026-10-08:
//!
//! - compress_ratios: 43 entries, first 40 are 0,0 then 2 through L19 and 1
//!   from L20 (the trailing three are ignored, `core_validate.c:14-46`)
//! - kv_source_layers [2, 8, 14, 20], index_source_layers
//!   [2, 8, 14, 20, 24, 28, 32, 36], engram.layer_ids [1, 14]
//! - mtp: 3 towers, 128 experts
//!
//! The tensor list is the artifact's published 1000 names
//! (`fixtures/v41/artifact-names.txt`), so the catalog gate is exact set
//! equality in both directions, not a spot check.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use ds4_core::{
    bind_names_v41, expected_layouts_v41, shape_for_variant, validate_layouts_v41, BindNeed,
    BindPlan, GgufFile, Shape, TensorInfo, TensorInventory, V41Wire, V41WireError, Variant,
};

fn shape() -> Shape {
    shape_for_variant(Variant::DeepSeek41Flash)
}

// The artifact's metadata arrays (see the module comment).
const COMPRESS_RATIOS: &[i32] = &[
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
];
const KV_SOURCES: &[i32] = &[2, 8, 14, 20];
const INDEX_SOURCES: &[i32] = &[2, 8, 14, 20, 24, 28, 32, 36];
const ENGRAM_LAYERS: &[i32] = &[1, 14];

enum Val<'a> {
    U32(u32),
    U64(u64),
    Str(&'a str),
    ArrayI32(&'a [i32]),
    ArrayU64(&'a [u64]),
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_str(buf: &mut Vec<u8>, s: &str) {
    put_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn write_gguf(path: &Path, kvs: &[(&str, Val<'_>)]) {
    let mut buf = Vec::new();
    put_u32(&mut buf, 0x4655_4747);
    put_u32(&mut buf, 3);
    put_u64(&mut buf, 0);
    put_u64(&mut buf, kvs.len() as u64);
    for (key, val) in kvs {
        put_str(&mut buf, key);
        match val {
            Val::U32(v) => {
                put_u32(&mut buf, 4);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Val::U64(v) => {
                put_u32(&mut buf, 10);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Val::Str(s) => {
                put_u32(&mut buf, 8);
                put_str(&mut buf, s);
            }
            Val::ArrayI32(items) => {
                put_u32(&mut buf, 9);
                put_u32(&mut buf, 5);
                put_u64(&mut buf, items.len() as u64);
                for x in *items {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            }
            Val::ArrayU64(items) => {
                put_u32(&mut buf, 9);
                put_u32(&mut buf, 10);
                put_u64(&mut buf, items.len() as u64);
                for x in *items {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
    while buf.len() < 32 {
        buf.push(0);
    }
    fs::write(path, buf).unwrap();
}

/// Each call gets its own file: the GGUF reader mmaps it, and tests run in
/// parallel threads, so a shared name would truncate a live mapping (SIGBUS).
fn tmp(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join("ds4-v41");
    fs::create_dir_all(&dir).unwrap();
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    dir.join(format!("{n}-{name}"))
}

/// The artifact's metadata, with the arrays overridable for the refusal cases.
fn artifact_gguf(name: &str, kv: &[i32], idx: &[i32], ratios: &[i32]) -> GgufFile {
    artifact_gguf_mtp(
        name,
        kv,
        idx,
        ratios,
        Some(5),
        Some(3),
        Some(128_799),
        Some(256),
        Some(&[37, 38, 39]),
        Some(1_048_576),
    )
}

/// `artifact_gguf` with the draft keys overridable: the engine arms the
/// drafter only when all four scalars and a nonempty target list are present,
/// and requires the list once the scalars are complete
/// (`core_validate_v41.c:97-110`; a `None` drops that key).  `ctx` likewise:
/// `None` drops `deepseek4.context_length` (the engine's hard stop,
/// `core_validate_v41.c:51-53`).
#[allow(clippy::too_many_arguments)]
fn artifact_gguf_mtp(
    name: &str,
    kv: &[i32],
    idx: &[i32],
    ratios: &[i32],
    block: Option<u32>,
    used: Option<u32>,
    noise: Option<u32>,
    rank: Option<u32>,
    targets: Option<&[i32]>,
    ctx: Option<u64>,
) -> GgufFile {
    let path = tmp(name);
    let mut kvs: Vec<(&str, Val<'_>)> = vec![
        ("deepseek4.attention.compress_ratios", Val::ArrayI32(ratios)),
        ("deepseek4.attention.kv_source_layers", Val::ArrayI32(kv)),
        (
            "deepseek4.attention.index_source_layers",
            Val::ArrayI32(idx),
        ),
        ("deepseek4.engram.layer_ids", Val::ArrayI32(ENGRAM_LAYERS)),
        (
            "deepseek4.engram.num_embeddings",
            Val::ArrayU64(&[384_006_168, 384_016_682]),
        ),
        ("deepseek4.engram.max_ngram_size", Val::U32(4)),
        ("deepseek4.engram.head_count", Val::U32(8)),
        ("deepseek4.engram.head_dim", Val::U32(256)),
        ("deepseek4.engram.pad_id_compressed", Val::U32(2)),
        // The engram vocab pair: required keys (core_validate_v41.c:152-153),
        // artifact values read from the GGUF on 2026-10-10.
        ("deepseek4.engram.vocab_size", Val::U32(16_000_000)),
        ("deepseek4.engram.compressed_vocab_size", Val::U32(99_092)),
        // The artifact's candidate two-level topk (L20 screens, 2048
        // blocks of 8); required since the ABI carries the triple.
        ("deepseek4.attention.candidate.source_layer", Val::U32(20)),
        ("deepseek4.attention.candidate.topk_blocks", Val::U32(2048)),
        ("deepseek4.attention.candidate.block_size", Val::U32(8)),
        (
            "deepseek4.engram.0.table_path",
            Val::Str("/home/fodelf/ds4-main/hf/model-00047-of-00048.safetensors"),
        ),
        ("deepseek4.engram.0.weight_offset", Val::U64(664)),
        ("deepseek4.engram.0.scale_offset", Val::U64(98_305_579_672)),
        (
            "deepseek4.engram.1.table_path",
            Val::Str("/home/fodelf/ds4-main/hf/model-00048-of-00048.safetensors"),
        ),
        ("deepseek4.engram.1.weight_offset", Val::U64(664)),
        ("deepseek4.engram.1.scale_offset", Val::U64(98_305_852_864)),
        ("deepseek4.mtp.tower_count", Val::U32(3)),
        ("deepseek4.mtp.expert_count", Val::U32(128)),
    ];
    if let Some(v) = block {
        kvs.push(("deepseek4.mtp.block_size", Val::U32(v)));
    }
    if let Some(v) = used {
        kvs.push(("deepseek4.mtp.expert_used_count", Val::U32(v)));
    }
    if let Some(v) = noise {
        kvs.push(("deepseek4.mtp.noise_token_id", Val::U32(v)));
    }
    if let Some(v) = rank {
        kvs.push(("deepseek4.mtp.markov_rank", Val::U32(v)));
    }
    if let Some(t) = targets {
        kvs.push(("deepseek4.mtp.target_layers", Val::ArrayI32(t)));
    }
    if let Some(v) = ctx {
        kvs.push(("deepseek4.context_length", Val::U64(v)));
    }
    write_gguf(&path, &kvs);
    GgufFile::open(&path).unwrap()
}

fn artifact_wire() -> V41Wire {
    V41Wire::load(
        &artifact_gguf("artifact.gguf", KV_SOURCES, INDEX_SOURCES, COMPRESS_RATIOS),
        &shape(),
    )
    .unwrap()
}

/// A synthetic inventory holding exactly `names`, one tensor each. Only the
/// name matters to the bind plan; offsets and dims are placeholders.
fn inventory_from(names: &[String]) -> TensorInventory {
    let tensors = names
        .iter()
        .enumerate()
        .map(|(i, name)| TensorInfo {
            name: name.clone(),
            ndim: 1,
            dim: [1, 0, 0, 0, 0, 0, 0, 0],
            typ: 0,
            rel_offset: i as u64 * 4,
            abs_offset: i as u64 * 4,
            elements: 1,
            bytes: 4,
            shard: 0,
        })
        .collect();
    TensorInventory {
        shards: Vec::new(),
        tensors,
        data_pos: 0,
        alignment: 32,
        page: 4096,
    }
}

fn artifact_names() -> Vec<String> {
    let raw = include_str!("fixtures/v41/artifact-names.txt");
    raw.lines().map(|s| s.to_string()).collect()
}

/// The artifact's real tensor directory (name, type, dims), dumped from the
/// GGUF on 2026-10-08. This is the layout gate's ground truth: every spec
/// must match the type and dims the artifact actually carries.
fn artifact_inventory() -> TensorInventory {
    let raw = include_str!("fixtures/v41/artifact-tensors.txt");
    let tensors = raw
        .lines()
        .map(|line| {
            let mut f = line.split('\t');
            let name = f.next().unwrap().to_string();
            let typ: u32 = f.next().unwrap().parse().unwrap();
            let dims: Vec<u64> = f
                .next()
                .unwrap()
                .split('x')
                .map(|d| d.parse().unwrap())
                .collect();
            let mut dim = [0u64; 8];
            dim[..dims.len()].copy_from_slice(&dims);
            TensorInfo {
                name,
                ndim: dims.len() as u32,
                dim,
                typ,
                rel_offset: 0,
                abs_offset: 0,
                elements: dims.iter().product(),
                bytes: 0,
                shard: 0,
            }
        })
        .collect();
    TensorInventory {
        shards: Vec::new(),
        tensors,
        data_pos: 0,
        alignment: 32,
        page: 4096,
    }
}

#[test]
fn wire_matches_the_artifacts_wiring() {
    let w = artifact_wire();
    // P5.4: the context length and the engram vocab pair (artifact values
    // read from the GGUF on 2026-10-10).
    assert_eq!(w.ctx, 1_048_576);
    assert_eq!(w.engram_vocab, 16_000_000);
    assert_eq!(w.engram_cvocab, 99_092);
    assert_eq!(w.mtp_towers, 3);
    assert_eq!(w.mtp_experts, 128);
    // P5: the draft parameters, as the artifact carries them.
    assert_eq!(w.mtp_block, Some(5));
    assert_eq!(w.mtp_used, Some(3));
    assert_eq!(w.mtp_noise_id, Some(128_799));
    assert_eq!(w.mtp_markov_rank, Some(256));
    assert_eq!(w.mtp_targets, vec![37, 38, 39]);
    assert!(w.mtp_armed());
    assert_eq!(w.engram_layers, vec![1, 14]);
    // Sources: the nearest earlier kv/index source per compressing layer.
    assert_eq!(w.kv_source_of[2], 2);
    assert_eq!(w.kv_source_of[7], 2);
    assert_eq!(w.kv_source_of[19], 14);
    assert_eq!(w.kv_source_of[24], 20);
    assert_eq!(w.kv_source_of[39], 20);
    assert_eq!(w.index_source_of[24], 24);
    assert_eq!(w.index_source_of[27], 24);
    assert_eq!(w.index_source_of[39], 36);
    // Ratio-0 layers compress nothing and read nothing.
    assert_eq!(w.kv_source_of[0], -1);
    assert_eq!(w.kv_source_of[1], -1);
    assert_eq!(w.index_source_of[1], -1);
    assert_eq!(w.engram_index_of[1], 0);
    assert_eq!(w.engram_index_of[14], 1);
    assert_eq!(w.engram_index_of[13], -1);
    // The engram table: two layers, two shard paths, two row planes each.
    assert_eq!(w.engram_rows, vec![384_006_168, 384_016_682]);
    assert_eq!(w.engram_weight_off[0], 664);
    assert_eq!(w.engram_pad, 2);
    assert!(w.engram_table_path[0].ends_with("model-00047-of-00048.safetensors"));
    // The tower experts are the blob form in this artifact.
    let inv = inventory_from(&artifact_names());
    assert!(w.tower_uses_blob(&inv, 0));
    assert!(w.tower_uses_blob(&inv, 2));
}

fn wire_mtp_block(block: u32) -> Result<V41Wire, V41WireError> {
    let g = artifact_gguf_mtp(
        "mtp-block.gguf",
        KV_SOURCES,
        INDEX_SOURCES,
        COMPRESS_RATIOS,
        Some(block),
        Some(3),
        Some(128_799),
        Some(256),
        Some(&[37, 38, 39]),
        Some(1_048_576),
    );
    V41Wire::load(&g, &shape())
}

#[test]
fn wire_mtp_block_bounds() {
    assert!(wire_mtp_block(9).is_err(), "accepted block 9");
}

#[test]
fn wire_rejects_mtp_block_wrap() {
    assert!(wire_mtp_block(u32::MAX).is_err(), "accepted wrapping block");
}

#[test]
fn wire_mtp_block_accepts() {
    for block in [1, 5, 8] {
        let wire = wire_mtp_block(block).expect("bounded draft block");
        assert_eq!(wire.mtp_block, Some(block));
        assert!(wire.mtp_armed());
    }
}

#[test]
fn zero_mtp_block_is_unarmed() {
    let wire = wire_mtp_block(0).expect("zero block retains unarmed compatibility");
    assert_eq!(wire.mtp_block, Some(0));
    assert!(!wire.mtp_armed());
}

#[test]
fn draft_parameters_arm_only_when_complete() {
    // A GGUF converted before the draft parameters existed: the towers bind
    // but speculation stays off (the engine warns and decodes one token at a
    // time; core_validate_v41.c:105-110).  The list still parses -- layout
    // needs it for the main_proj shape.
    let g = artifact_gguf_mtp(
        "mtp-unarmed.gguf",
        KV_SOURCES,
        INDEX_SOURCES,
        COMPRESS_RATIOS,
        None,
        Some(3),
        Some(128_799),
        Some(256),
        Some(&[37, 38, 39]),
        Some(1_048_576),
    );
    let w = V41Wire::load(&g, &shape()).unwrap();
    assert!(!w.mtp_armed());
    assert_eq!(w.mtp_block, None);
    assert_eq!(w.mtp_targets, vec![37, 38, 39]);

    // With the scalars complete the target list is required, not optional:
    // the engine exits when it is missing (v41_arr_i32), never guesses.
    let g = artifact_gguf_mtp(
        "mtp-no-targets.gguf",
        KV_SOURCES,
        INDEX_SOURCES,
        COMPRESS_RATIOS,
        Some(5),
        Some(3),
        Some(128_799),
        Some(256),
        None,
        Some(1_048_576),
    );
    let err = V41Wire::load(&g, &shape()).expect_err("scalars without targets");
    assert_eq!(err, V41WireError::MissingKey("deepseek4.mtp.target_layers"));

    // An empty target list parses but never arms.
    let g = artifact_gguf_mtp(
        "mtp-empty-targets.gguf",
        KV_SOURCES,
        INDEX_SOURCES,
        COMPRESS_RATIOS,
        Some(5),
        Some(3),
        Some(128_799),
        Some(256),
        Some(&[]),
        Some(1_048_576),
    );
    let w = V41Wire::load(&g, &shape()).unwrap();
    assert!(!w.mtp_armed());

    // An over-cap target list is the engine's hard stop (v41_arr_i32).
    let over: Vec<i32> = (0..9).collect();
    let g = artifact_gguf_mtp(
        "mtp-over-targets.gguf",
        KV_SOURCES,
        INDEX_SOURCES,
        COMPRESS_RATIOS,
        Some(5),
        Some(3),
        Some(128_799),
        Some(256),
        Some(&over),
        Some(1_048_576),
    );
    let err = V41Wire::load(&g, &shape()).expect_err("over-cap targets");
    assert_eq!(err, V41WireError::ArrayLen("deepseek4.mtp.target_layers"));
}

#[test]
fn context_length_is_required_and_range_checked() {
    // The engine's only context source (core_validate_v41.c:51-53): a
    // missing key is a hard stop, and 0 or over-u32::MAX is invalid.  The
    // Rust host must refuse here because the native generate entry and the
    // state alloc treat a zero ctx as "no metadata" and refuse too.
    let g = artifact_gguf_mtp(
        "ctx-missing.gguf",
        KV_SOURCES,
        INDEX_SOURCES,
        COMPRESS_RATIOS,
        Some(5),
        Some(3),
        Some(128_799),
        Some(256),
        Some(&[37, 38, 39]),
        None,
    );
    let err = V41Wire::load(&g, &shape()).expect_err("no context_length");
    assert_eq!(err, V41WireError::MissingKey("deepseek4.context_length"));

    let g = artifact_gguf_mtp(
        "ctx-zero.gguf",
        KV_SOURCES,
        INDEX_SOURCES,
        COMPRESS_RATIOS,
        Some(5),
        Some(3),
        Some(128_799),
        Some(256),
        Some(&[37, 38, 39]),
        Some(0),
    );
    let err = V41Wire::load(&g, &shape()).expect_err("zero context_length");
    assert_eq!(err, V41WireError::CtxInvalid);

    let g = artifact_gguf_mtp(
        "ctx-over.gguf",
        KV_SOURCES,
        INDEX_SOURCES,
        COMPRESS_RATIOS,
        Some(5),
        Some(3),
        Some(128_799),
        Some(256),
        Some(&[37, 38, 39]),
        Some(u32::MAX as u64 + 1),
    );
    let err = V41Wire::load(&g, &shape()).expect_err("over-u32 context_length");
    assert_eq!(err, V41WireError::CtxInvalid);
}

#[test]
fn engram_dir_override_keeps_only_the_file_name() {
    let mut w = artifact_wire();
    w.apply_engram_dir("/srv/engram");
    assert_eq!(
        w.engram_table_path[0],
        "/srv/engram/model-00047-of-00048.safetensors"
    );
    assert_eq!(
        w.engram_table_path[1],
        "/srv/engram/model-00048-of-00048.safetensors"
    );
}

#[test]
fn compressing_layer_without_a_source_is_refused() {
    let g = artifact_gguf("no-source.gguf", &[], &[], COMPRESS_RATIOS);
    let err = V41Wire::load(&g, &shape()).expect_err("ratio 2 with no kv source");
    assert_eq!(err, V41WireError::NoSourceBefore(2));
}

#[test]
fn kv_source_must_also_be_an_index_source() {
    // L4 is a kv source that never appears in index_source_layers.
    let mut ratios = vec![0i32; 40];
    ratios[4] = 1;
    let g = artifact_gguf("kv-not-index.gguf", &[2, 4], &[2], &ratios);
    let err = V41Wire::load(&g, &shape()).expect_err("kv source without index source");
    assert_eq!(err, V41WireError::KvSourceNotIndexSource(4));
}

#[test]
fn a_layer_id_outside_the_shape_is_refused() {
    let g = artifact_gguf("layer-range.gguf", &[2, 40], INDEX_SOURCES, COMPRESS_RATIOS);
    let err = V41Wire::load(&g, &shape()).expect_err("layer 40 in a 40-layer model");
    assert_eq!(
        err,
        V41WireError::LayerRange("deepseek4.attention.kv_source_layers", 40)
    );
}

#[test]
fn a_non_v41_shape_is_refused() {
    let g = artifact_gguf("not-v41.gguf", KV_SOURCES, INDEX_SOURCES, COMPRESS_RATIOS);
    let err = V41Wire::load(&g, &shape_for_variant(Variant::Flash)).expect_err("not v4.1");
    assert_eq!(err, V41WireError::NotV41);
}

#[test]
fn bind_catalog_equals_the_artifact_tensor_set() {
    let wire = artifact_wire();
    let names = artifact_names();
    let inv = inventory_from(&names);
    let catalog = bind_names_v41(&shape(), &wire, &inv);

    let catalog_set: HashSet<&str> = catalog.iter().map(|n| n.name.as_str()).collect();
    let artifact_set: HashSet<&str> = names.iter().map(|s| s.as_str()).collect();
    let missing: Vec<_> = artifact_set.difference(&catalog_set).collect();
    let extra: Vec<_> = catalog_set.difference(&artifact_set).collect();
    assert!(
        missing.is_empty(),
        "artifact tensors not in catalog: {missing:?}"
    );
    assert!(extra.is_empty(), "catalog names not in artifact: {extra:?}");
    assert_eq!(catalog.len(), 1000);
    assert!(catalog.iter().all(|n| n.need == BindNeed::Required));

    let plan = BindPlan::resolve_v41(shape(), &wire, &inv);
    assert_eq!(plan.missing_required(), Vec::<&str>::new());
}

#[test]
fn per_expert_towers_swap_the_blob_for_every_expert() {
    let wire = artifact_wire();
    let mut names = artifact_names();
    names.retain(|n| !n.ends_with("ffn_exps_vq.blob") || !n.starts_with("mtp."));
    for t in 0..3 {
        for e in 0..128 {
            for suffix in ["gate.weight", "up.weight", "down.weight"] {
                names.push(format!("mtp.{t}.ffn_exp.{e}.{suffix}"));
            }
        }
    }
    let inv = inventory_from(&names);
    let catalog = bind_names_v41(&shape(), &wire, &inv);
    assert!(!catalog.iter().any(|n| n.name == "mtp.0.ffn_exps_vq.blob"));
    assert!(catalog
        .iter()
        .any(|n| n.name == "mtp.2.ffn_exp.127.down.weight"));
    let plan = BindPlan::resolve_v41(shape(), &wire, &inv);
    assert_eq!(plan.missing_required(), Vec::<&str>::new());
}

#[test]
fn a_tower_with_neither_expert_form_is_refused() {
    let wire = artifact_wire();
    let mut names = artifact_names();
    names.retain(|n| !n.starts_with("mtp."));
    let inv = inventory_from(&names);
    let plan = BindPlan::resolve_v41(shape(), &wire, &inv);
    let missing = plan.missing_required();
    assert!(missing.contains(&"mtp.0.ffn_exp.0.gate.weight"));
    assert!(missing.contains(&"mtp.main_proj.weight"));
}

#[test]
fn layout_matches_the_artifact_tensor_types_and_dims() {
    let wire = artifact_wire();
    let inv = artifact_inventory();
    let plan = BindPlan::resolve_v41(shape(), &wire, &inv);
    assert_eq!(plan.missing_required(), Vec::<&str>::new());
    validate_layouts_v41(&plan, &wire, &inv).expect("layout matches the artifact");

    // Spec coverage must equal the plan exactly; expect_specs only walks the
    // specs, so an extra plan name would otherwise go unchecked.
    let specs = expected_layouts_v41(&shape(), &wire, &inv);
    let spec_names: HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    let plan_names: HashSet<&str> = plan.slots.iter().map(|s| s.name.as_str()).collect();
    let without_spec: Vec<_> = plan_names.difference(&spec_names).collect();
    let without_slot: Vec<_> = spec_names.difference(&plan_names).collect();
    assert!(
        without_spec.is_empty(),
        "plan names without a layout spec: {without_spec:?}"
    );
    assert!(
        without_slot.is_empty(),
        "layout specs without a plan name: {without_slot:?}"
    );
    assert_eq!(specs.len(), 1000);
}

#[test]
fn a_wrong_type_or_dim_is_refused() {
    let wire = artifact_wire();
    let mut inv = artifact_inventory();
    // The artifact's attn_q_a is q4_K; fp8_32x32 is not in the skeleton set.
    let t = inv
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.attn_q_a.weight")
        .unwrap();
    t.typ = 44;
    let plan = BindPlan::resolve_v41(shape(), &wire, &inv);
    let err = validate_layouts_v41(&plan, &wire, &inv).expect_err("fp8 skeleton");
    assert_eq!(err.token(), "type blk.0.attn_q_a.weight");

    let mut inv = artifact_inventory();
    let t = inv
        .tensors
        .iter_mut()
        .find(|t| t.name == "blk.5.ffn_exps_vq.blob")
        .unwrap();
    t.ndim = 2;
    let plan = BindPlan::resolve_v41(shape(), &wire, &inv);
    let err = validate_layouts_v41(&plan, &wire, &inv).expect_err("blob ndim");
    assert_eq!(err.token(), "ndim blk.5.ffn_exps_vq.blob");
}
