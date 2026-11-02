//! Gemma-4-26B-A4B VQ registration. The shape is selectable by name, the
//! serving surface refuses every lane instead of quoting one, and the GGUF
//! path refuses the family before it can bind anything.
//!
//! The artifact-backed tests read JSON and file headers only (no GPU, no
//! inference); point `DS4_GEMMA4_MODEL` at the release directory to run them.

use std::fs;
use std::path::{Path, PathBuf};

use ds4_core::gemma4_a4b::{
    decode, is_release, matvec, Codebook, Config, Entry, LayerType, Payload,
};
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

// ---- decode -------------------------------------------------------------

const RQVQ: &[u8; 4] = b"RQVQ";
const DQV3: &[u8; 4] = b"DQV3";
const RQVF: &[u8; 4] = b"RQVF";

fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// f32 -> f16 for values that are exact in half precision; asserts exactness.
fn half(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xFF) as i32 - 127 + 15) as u16;
    let man = ((bits >> 13) & 0x3FF) as u16;
    assert!(exp > 0 && exp < 31, "value not a normal half");
    let h = sign | (exp << 10) | man;
    assert_eq!(ds4_core::f16_to_f32(h), f, "value not exact in half");
    h
}

/// The fixture's codeword: `(w + 1) * (d + 1)`, exact in f16.
fn cb(w: u16, d: u16) -> f32 {
    (w as f32 + 1.0) * (d as f32 + 1.0)
}

fn index_bits(nc: u16) -> u32 {
    let mut n = 1u32;
    while (1u32 << n) < u32::from(nc) {
        n += 1;
    }
    n
}

/// One payload in the VQ tools' `DQV3` layout: 24-byte header, inline f16
/// codebook, gains, a row-local `min(nbit, 12)`-bit main stream, no plane.
/// `/data/VQ_tools/c/gpu/build-vq-gpu.c:82-126`.
fn tools_v3_payload(
    dim: u16,
    nc: u16,
    rows: u32,
    cols: u32,
    gains: &[f32],
    idx: &[Vec<u16>],
) -> Vec<u8> {
    let mb = index_bits(nc).min(12) as usize;
    let nidx_row = (cols / u32::from(dim)) as usize;
    let mrow = (nidx_row * mb + 7) / 8;
    let mut b = Vec::new();
    b.extend_from_slice(DQV3);
    put_u16(&mut b, dim);
    put_u16(&mut b, nc);
    put_u32(&mut b, rows);
    put_u32(&mut b, cols);
    put_u32(&mut b, 0); // flags: no 13th-bit plane
    put_u32(&mut b, 0); // pad, so the codebook lands 8-byte aligned
    for w in 0..nc {
        for d in 0..dim {
            put_u16(&mut b, half(cb(w, d)));
        }
    }
    for g in gains {
        put_u16(&mut b, half(*g));
    }
    let mut main = vec![0u8; rows as usize * mrow];
    for (r, row) in idx.iter().enumerate() {
        for (i, v) in row.iter().enumerate() {
            for bit in 0..mb {
                if (u32::from(*v) >> bit) & 1 == 1 {
                    let at = i * mb + bit;
                    main[r * mrow + (at >> 3)] |= 1 << (at & 7);
                }
            }
        }
    }
    b.extend_from_slice(&main);
    b
}

/// An `RQVQ` container: 32-byte header, then the payloads back to back.
fn rqvq(stages: u16, dim: u16, nc: u16, rows: u32, cols: u32, payloads: &[Vec<u8>]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(RQVQ);
    put_u16(&mut b, dim);
    put_u16(&mut b, nc);
    put_u16(&mut b, stages);
    put_u16(&mut b, 0);
    put_u32(&mut b, rows);
    put_u32(&mut b, cols);
    put_u32(&mut b, 0); // codebook format: f16
    let payload: usize = payloads.iter().map(Vec::len).sum();
    put_u64(&mut b, payload as u64);
    for p in payloads {
        b.extend_from_slice(p);
    }
    b
}

fn entry(name: &str, rows: usize, cols: usize) -> Entry {
    Entry {
        name: name.to_string(),
        part: "part".to_string(),
        role: "role".to_string(),
        encoded: true,
        rows,
        cols,
        dim: Some(4),
        nc: Some(4),
        stages: Some(2),
        groups: Some(1),
        codebook: Some(Codebook::F16),
        payload: Some(Payload::V3),
        bytes: 0,
        sha256: "0".repeat(64),
    }
}

/// The fixture both decode tests use: dim 4, nc 4, 2 rows of 8 columns,
/// two residual stages with their own gains.
fn fixture() -> (Vec<u8>, Vec<Vec<u16>>, Vec<Vec<u16>>, [f32; 2], [f32; 2]) {
    let (dim, nc, rows, cols) = (4u16, 4u16, 2u32, 8u32);
    let idx0 = vec![vec![1u16, 2], vec![3, 0]];
    let idx1 = vec![vec![0u16, 3], vec![2, 1]];
    let g0 = [1.0f32, 0.5];
    let g1 = [2.0f32, 0.25];
    let p0 = tools_v3_payload(dim, nc, rows, cols, &g0, &idx0);
    let p1 = tools_v3_payload(dim, nc, rows, cols, &g1, &idx1);
    let file = rqvq(2, dim, nc, rows, cols, &[p0, p1]);
    (file, idx0, idx1, g0, g1)
}

fn write_entry(dir: &Path, e: &mut Entry, file: &[u8]) {
    fs::write(dir.join(format!("{}.bin", e.name)), file).unwrap();
    e.bytes = file.len() as u64;
}

#[test]
fn a_tools_v3_container_decodes_and_its_stages_sum() {
    let dir = scratch("decode-v3");
    let (file, idx0, idx1, g0, g1) = fixture();
    let mut e = entry("model.language_model.layers.0.weight", 2, 8);
    write_entry(&dir, &mut e, &file);

    let t = decode(&dir, &e).unwrap();
    assert_eq!((t.rows, t.cols), (2, 8));
    for r in 0..2usize {
        for i in 0..2usize {
            for d in 0..4u16 {
                let want = cb(idx0[r][i], d) * g0[r] + cb(idx1[r][i], d) * g1[r];
                assert_eq!(
                    t.at(r, i * 4 + d as usize),
                    want,
                    "row {r} index {i} dim {d}"
                );
            }
        }
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_container_header_and_the_payload_layout_are_both_checked() {
    let dir = scratch("decode-refuse");
    let (file, _, _, _, _) = fixture();
    let mut e = entry("model.language_model.layers.0.weight", 2, 8);
    write_entry(&dir, &mut e, &file);

    // The release's geometry must agree with the container header.
    let mut wrong = e.clone();
    wrong.stages = Some(3);
    let error = decode(&dir, &wrong).unwrap_err().to_string();
    assert!(error.contains("header stages is 2"), "{error}");

    // A payload that states another dim than the container is refused by the
    // payload's own header, not decoded under the container's guess.
    let mut other_dim = file.clone();
    other_dim[32 + 4] = 8;
    let mut e2 = e.clone();
    write_entry(&dir, &mut e2, &other_dim);
    let error = decode(&dir, &e2).unwrap_err().to_string();
    assert!(error.contains("vq-payload-dim 8"), "{error}");

    // An engine-layout v3 payload (mnb 12 where the tools write the pad word)
    // is the other producer's format and must not be decoded as this one.
    let mut engine = file.clone();
    engine[32 + 20..32 + 24].copy_from_slice(&12u32.to_le_bytes());
    let mut e3 = e.clone();
    write_entry(&dir, &mut e3, &engine);
    let error = decode(&dir, &e3).unwrap_err().to_string();
    assert!(error.contains("vq-payload-flags"), "{error}");

    // The recorded payload kind must match the first payload's magic.
    let mut wrong_kind = e.clone();
    wrong_kind.payload = Some(Payload::V2);
    let error = decode(&dir, &wrong_kind).unwrap_err().to_string();
    assert!(error.contains("payload kind"), "{error}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_copied_sidecar_is_read_as_f32() {
    let dir = scratch("decode-rqvf");
    let name = "model.language_model.layers.0.input_layernorm.weight";
    let mut file = Vec::new();
    file.extend_from_slice(RQVF);
    put_u32(&mut file, 1);
    put_u32(&mut file, 3);
    for v in [0.5f32, -1.5, 2.0] {
        file.extend_from_slice(&v.to_le_bytes());
    }
    let mut e = entry(name, 1, 3);
    e.encoded = false;
    e.dim = None;
    e.nc = None;
    e.stages = Some(0);
    e.groups = None;
    e.codebook = None;
    e.payload = None;
    write_entry(&dir, &mut e, &file);

    let t = decode(&dir, &e).unwrap();
    assert_eq!((t.rows, t.cols), (1, 3));
    assert_eq!(t.data, vec![0.5, -1.5, 2.0]);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_fused_dot_matches_the_decoded_row() {
    let dir = scratch("dot");
    let (file, _, _, _, _) = fixture();
    let mut e = entry("model.language_model.layers.0.weight", 2, 8);
    write_entry(&dir, &mut e, &file);
    let per = (file.len() - 32) / 2;
    let m0 = ds4_core::VqMatrix::from_payload(&file[32..32 + per], 2, 4, 2, 8).unwrap();
    let m1 = ds4_core::VqMatrix::from_payload(&file[32 + per..], 2, 4, 2, 8).unwrap();
    let t = decode(&dir, &e).unwrap();

    let x = [0.5f32, -1.0, 2.0, 0.25, 1.5, -0.5, 0.75, 3.0];
    for r in 0..2usize {
        // The reference order: the decoded row, column by column.
        let mut want = 0.0f32;
        for (c, value) in x.iter().enumerate() {
            want += t.at(r, c) * value;
        }
        let got = m0.dot_row(r, &x) + m1.dot_row(r, &x);
        assert_eq!(got.to_bits(), want.to_bits(), "row {r}");
    }

    // The matrix form accumulates every payload at its group's first row.
    let all = matvec(&dir, &e, &x).unwrap();
    assert_eq!(all.len(), 2);
    for r in 0..2usize {
        assert_eq!(
            all[r].to_bits(),
            (m0.dot_row(r, &x) + m1.dot_row(r, &x)).to_bits()
        );
    }
    fs::remove_dir_all(&dir).unwrap();
}

/// The builder's own relL2 column, in percent (`c/gpu/build-vq-gpu.c:544-552`).
fn manifest_rel_l2(dir: &Path, name: &str) -> f64 {
    let text = fs::read_to_string(dir.join("manifest.txt")).unwrap();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() >= 6 && f[0] == name {
            return f[5].parse().unwrap();
        }
    }
    panic!("{name} is not in manifest.txt");
}

fn source_f32(dir: &Path, name: &str) -> Vec<f32> {
    use ds4_core::minimax_h3::safetensors::{self, Dtype};
    let store = safetensors::open_dir(dir).unwrap();
    let (shard, tensor) = store
        .find(name)
        .unwrap_or_else(|| panic!("{name} in the source"));
    let mut raw = vec![0u8; tensor.bytes() as usize];
    shard.read_data(tensor, &mut raw).unwrap();
    match tensor.dtype {
        Dtype::Bf16 => raw
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| ds4_core::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("{name}: source dtype {other:?} is not handled"),
    }
}

#[test]
#[ignore = "requires DS4_GEMMA4_MODEL and DS4_GEMMA4_SOURCE; decodes two real tensors, no GPU"]
fn the_reference_tensors_decode_to_the_source() {
    let dir = PathBuf::from(std::env::var_os("DS4_GEMMA4_MODEL").expect("DS4_GEMMA4_MODEL"));
    let src = PathBuf::from(std::env::var_os("DS4_GEMMA4_SOURCE").expect("DS4_GEMMA4_SOURCE"));
    let family = ds4_core::gemma4_a4b::Family::open(&dir).unwrap();
    let mut checked = 0;
    for name in [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.router.proj.weight",
    ] {
        let e = family.release.get(name).expect("release entry");
        let got = decode(&dir, e).unwrap();
        let want = source_f32(&src, name);
        assert_eq!(got.data.len(), want.len(), "{name}: element count");
        let mut se = 0f64;
        let mut sn = 0f64;
        for (a, b) in got.data.iter().zip(&want) {
            let d = f64::from(*a) - f64::from(*b);
            se += d * d;
            sn += f64::from(*b) * f64::from(*b);
        }
        let rel = 100.0 * (se / (sn + 1e-30)).sqrt();
        let recorded = manifest_rel_l2(&dir, name);
        println!(
            "GEMMA4_DECODE {name} {}x{} dim {} relL2 {rel:.4}% manifest {recorded:.4}%",
            got.rows,
            got.cols,
            e.dim.map(|d| d.to_string()).unwrap_or_default()
        );
        assert!(
            (rel - recorded).abs() < 0.01,
            "{name}: decoded relL2 {rel:.4}%, the builder recorded {recorded:.4}%"
        );

        // The fused dot must agree with the dot of the decoded row, because
        // the forward never materialises a matrix. The two orders reassociate
        // the sum (the gain is applied after the dot), so this is a measured
        // bound, not bit equality; the fixture test proves bit equality where
        // the values are exact.
        let x: Vec<f32> = (0..got.cols)
            .map(|c| ((c % 17) as f32 - 8.0) * 0.125)
            .collect();
        let fused = matvec(&dir, e, &x).unwrap();
        let mut max_abs = 0f32;
        let mut scale = 0f32;
        for r in 0..got.rows {
            let mut want = 0.0f32;
            for (c, value) in x.iter().enumerate() {
                want += got.at(r, c) * value;
            }
            scale = scale.max(want.abs());
            max_abs = max_abs.max((fused[r] - want).abs());
        }
        println!(
            "GEMMA4_MATVEC {name} rows={} max_abs={max_abs:e} scale={scale:e}",
            got.rows
        );
        assert!(
            max_abs <= 1e-4 * scale.max(1.0),
            "{name}: fused dot differs by {max_abs:e}, row scale {scale:e}"
        );
        checked += 1;
    }
    assert_eq!(checked, 2);
}
