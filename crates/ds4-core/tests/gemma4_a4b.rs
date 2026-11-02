//! Gemma-4-26B-A4B VQ registration. The shape is selectable by name, the
//! serving surface refuses every lane instead of quoting one, and the GGUF
//! path refuses the family before it can bind anything.
//!
//! The artifact-backed tests read JSON and file headers only (no GPU, no
//! inference); point `DS4_GEMMA4_MODEL` at the release directory to run them.

use std::fs;
use std::path::{Path, PathBuf};

use ds4_core::gemma4_a4b::{is_release, Config, LayerType};
use ds4_core::{
    route_architecture, shape_for_variant, ArchRoute, HostNeed, ModelFamily, ReuseKind, Shape,
    SpecLane, Support, Variant,
};

fn shape() -> Shape {
    shape_for_variant(Variant::Gemma4A4bVq)
}

#[test]
fn family_is_its_own_contract() {
    let shape = shape();
    assert_eq!(shape.family, ModelFamily::Gemma4A4b);
    assert_eq!(shape.variant, Variant::Gemma4A4bVq);
    assert_eq!(shape.name, "Gemma-4-26B-A4B-VQ");
    assert_eq!(
        (shape.n_layer, shape.n_embd, shape.n_vocab),
        (30, 2816, 262_144)
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
    // The un-prefixed geometry is the full-attention one, the convention
    // `layout.rs` follows when it chooses `n_head` or `n_swa_head`.
    assert_eq!(
        (shape.n_head, shape.n_head_dim, shape.n_head_kv, shape.n_rot),
        (16, 512, 2, 128)
    );
    assert_eq!(
        (shape.n_swa, shape.n_swa_period, shape.n_full_attn_count),
        (1024, 0, 5)
    );
    assert_eq!(shape.n_swa_head, 16);
    assert!((shape.rope_freq_base - 1_000_000.0).abs() < 1.0);
    assert!((shape.rope_freq_base_swa - 10_000.0).abs() < 1.0);
    assert!((shape.rms_eps - 1e-6).abs() < 1e-12);
    assert!(shape.use_qk_norm);
}

#[test]
fn the_flat_shape_and_the_config_state_the_same_two_geometries() {
    let shape = shape();
    let config = Config::embedded().unwrap();
    assert_eq!(shape.n_layer as usize, config.layers);
    assert_eq!(shape.n_embd as usize, config.hidden);
    assert_eq!(shape.n_vocab as usize, config.vocab);
    assert_eq!(shape.n_head as usize, config.heads);
    assert_eq!(shape.n_expert as usize, config.experts);
    assert_eq!(shape.n_expert_used as usize, config.top_k);
    assert_eq!(shape.n_ff_exp as usize, config.moe_ff);
    assert_eq!(shape.n_ff_dense as usize, config.dense_ff);

    // The fields the full geometry owns.
    let full = LayerType::Full;
    assert_eq!(shape.n_head_dim as usize, config.head_dim_of(full));
    assert_eq!(shape.n_head_kv as usize, config.kv_heads_of(full));
    assert_eq!(shape.n_rot as usize, config.rotated_dims(full));
    assert!((shape.rope_freq_base as f64 - config.rope_theta(full)).abs() < 1.0);

    // The fields the sliding geometry owns.
    let sliding = LayerType::Sliding;
    assert_eq!(shape.n_swa as usize, config.sliding_window);
    assert_eq!(shape.n_swa_head as usize, config.heads);
    assert!((shape.rope_freq_base_swa as f64 - config.rope_theta(sliding)).abs() < 1.0);

    // The two geometries really are different, and the flat shape cannot state
    // the sliding head width or KV head count: no field holds them, so the
    // configuration stays the only authority for those.
    assert_ne!(config.head_dim_of(sliding), config.head_dim_of(full));
    assert_ne!(config.kv_heads_of(sliding), config.kv_heads_of(full));
    assert_eq!(
        config
            .layer_types
            .iter()
            .filter(|kind| **kind == LayerType::Full)
            .count() as u32,
        shape.n_full_attn_count
    );
    assert_eq!(
        config
            .layer_types
            .iter()
            .filter(|kind| config.has_v_proj(**kind))
            .count(),
        25
    );
    let full_layers: Vec<usize> = (0..config.layers)
        .filter(|index| config.layer_type(*index) == LayerType::Full)
        .collect();
    assert_eq!(full_layers, vec![5, 11, 17, 23, 29]);
    assert_eq!(config.layer_type(0), LayerType::Sliding);
    assert_eq!(shape.n_swa_period, 0, "the schedule is not periodic");
}

#[test]
fn every_serving_lane_is_absent_and_the_context_is_the_native_maximum() {
    let caps = ds4_core::caps_from_shape(shape());
    assert_eq!(caps.ctx_max, Some(262_144));
    assert_eq!(caps.host, HostNeed::Cuda);
    assert_eq!(caps.bank_support, Support::None);
    assert_eq!(caps.reuse, ReuseKind::None);
    assert_eq!(caps.reuse_support, Support::None);
    assert_eq!(caps.disk, Support::None);
    assert_eq!(caps.snapshot, Support::None);
    assert_eq!(caps.mtp_support, Support::None);
    assert_eq!(caps.spec_lane, SpecLane::None);
    // Nothing is qualified: the family has no executor to qualify.
    assert_eq!(caps.qualified_ctx, None);
    assert_eq!(caps.qualified_banks, None);
    assert_eq!(caps.qualified_prompt, None);
}

#[test]
fn the_family_is_selectable_by_its_oracle_name() {
    assert_eq!(
        ModelFamily::from_oracle_name("gemma4_a4b"),
        Some(ModelFamily::Gemma4A4b)
    );
    assert_eq!(ModelFamily::Gemma4A4b.oracle_name(), "gemma4_a4b");
    assert!(matches!(
        route_architecture(Some(b"gemma4_a4b")),
        ArchRoute::Fixed(Variant::Gemma4A4bVq)
    ));
    // The discriminator is fresh, not a reuse of another family's.
    assert_eq!(ModelFamily::Gemma4A4b as u32, 19);
    assert_eq!(Variant::Gemma4A4bVq as u32, 23);
}

/// A GGUF carrying only the architecture key: enough for identification to
/// route to the family, so the refusal under test is the family's own.
fn write_arch_only_gguf(path: &Path) {
    fn put_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn put_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let mut buf = Vec::new();
    put_u32(&mut buf, 0x4655_4747);
    put_u32(&mut buf, 3);
    put_u64(&mut buf, 0);
    put_u64(&mut buf, 1);
    put_u64(&mut buf, "general.architecture".len() as u64);
    buf.extend_from_slice(b"general.architecture");
    put_u32(&mut buf, 8);
    put_u64(&mut buf, "gemma4_a4b".len() as u64);
    buf.extend_from_slice(b"gemma4_a4b");
    while buf.len() < 32 {
        buf.push(0);
    }
    fs::write(path, buf).unwrap();
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ds4-gemma4-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_gguf_path_refuses_the_family_by_name() {
    let dir = scratch("gguf");
    let path = dir.join("forced.gguf");
    write_arch_only_gguf(&path);
    let report = ds4_core::dump_validate(&path);
    assert!(
        report.contains("Gemma-4 VQ requires a complete release directory"),
        "{report}"
    );
    // The empty bind list is therefore unreachable on this path; pinned so a
    // silent zero-slot binder cannot appear later.
    let names = ds4_core::bind_names(&shape());
    assert!(names.is_empty());
    assert!(ds4_core::expected_layouts(&shape()).is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_release_is_claimed_only_by_this_family() {
    let dir = scratch("claim");
    let release = dir.join("vq-release.json");
    // No release at all.
    assert!(!is_release(&dir));
    // The schema alone is shared with every VQ family, so it is not enough.
    fs::write(&release, br#"{"schema":"vq-release/1","entries":[]}"#).unwrap();
    assert!(!is_release(&dir));
    // A GGUF-part release (the other families' shape) is not this one.
    fs::write(
        &release,
        br#"{"schema":"vq-release/1","entries":[{"name":"blk.0.attn_q.weight"}]}"#,
    )
    .unwrap();
    assert!(!is_release(&dir));
    // The first entry carries this artifact's own prefix.
    fs::write(
        &release,
        br#"{"schema":"vq-release/1","entries":[{"name":"model.language_model.embed_tokens.weight"}]}"#,
    )
    .unwrap();
    assert!(is_release(&dir));
    // Claimed but incomplete: identification reports this family's error
    // rather than falling through to another family's diagnosis.
    let error = ds4_core::identify_gguf(&dir).unwrap_err().to_string();
    assert!(error.starts_with("invalid-artifact"), "{error}");
    assert!(error.contains("release"), "{error}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore = "requires the release in DS4_GEMMA4_MODEL; verifies every digest, no GPU"]
fn the_release_identifies_and_matches_its_manifest() {
    let dir = PathBuf::from(std::env::var_os("DS4_GEMMA4_MODEL").expect("DS4_GEMMA4_MODEL"));
    let id = ds4_core::identify_gguf(&dir).unwrap();
    assert_eq!(id.shape.family, ModelFamily::Gemma4A4b);
    assert_eq!(id.n_tensors, 657);
    assert_eq!(id.split_count, 1);
    assert_eq!(ds4_core::caps_from_ident(&id).ctx_max, Some(262_144));
    let family = ds4_core::gemma4_a4b::Family::open(&dir).unwrap();
    family.verify().unwrap();
    println!(
        "GEMMA4_RELEASE_PASS entries={} ",
        family.release.totals.entries
    );
}
