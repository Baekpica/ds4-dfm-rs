//! V4.1 zchain merge gates, model-free: the ②×③ gain product, the rb table
//! coming from ② only, the amp rank concatenation with β on ② alone, the
//! `base.fnv` gate running before anything is read, and the refusals
//! (`core_v41_amp.c:123-270`).
//!
//! The live check - the real ② directory `…-vqfin41_vqhalf_a_n8192-engine`
//! against `posttrain-experimental-20260924` - runs the inspector on the
//! Spark; there layer 39 carries gains in both directories and the merged
//! factor is their product (port plan §6.6.4).

use std::fs;
use std::path::PathBuf;

use ds4_core::{gr_dir_fnv, BaseFingerprint, SidecarError, V41Zchain, ZchainGeom};

fn tmpdir(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join("ds4-zchain");
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = dir.join(format!("{n}-{name}"));
    // A previous run of this binary left the same paths behind; the tests
    // that assert emptiness need a genuinely fresh directory.
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

fn put_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// `gr_Lnn.bin` in the f32 form.
fn gr_f32(n_expert: i32, d: i32, vals: &[f32]) -> Vec<u8> {
    let mut b = Vec::new();
    put_i32(&mut b, n_expert);
    put_i32(&mut b, d);
    put_i32(&mut b, 1);
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// `gr_Lnn.bin` in the fp4x32 form, one 17-byte block per 32 elements. The
/// disk stores `s - 1`; the reader restores `1 + raw`.
fn gr_fp4(n_expert: i32, d: i32, blocks: &[[u8; 17]]) -> Vec<u8> {
    let mut b = Vec::new();
    put_i32(&mut b, n_expert);
    put_i32(&mut b, d);
    put_i32(&mut b, 43);
    for blk in blocks {
        b.extend_from_slice(blk);
    }
    b
}

fn rb(n_expert: i32, vals: &[f32]) -> Vec<u8> {
    let mut b = Vec::new();
    put_i32(&mut b, n_expert);
    put_i32(&mut b, 1);
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// `amp_Lnn.bin` in the f32 form: `<i32 D><i32 K><i32 1>`, then A[K][D], B[K][D].
fn amp_f32(d: i32, k: i32, a: &[f32], b: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    put_i32(&mut buf, d);
    put_i32(&mut buf, k);
    put_i32(&mut buf, 1);
    for v in a {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in b {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

const GEOM: ZchainGeom = ZchainGeom {
    n_layer: 4,
    n_expert: 4,
    n_embd: 2,
};

#[test]
fn naked_base_is_none_and_an_empty_plugin_refuses() {
    assert!(V41Zchain::load(None, None, GEOM, 1.0).unwrap().is_none());

    let empty = tmpdir("empty");
    match V41Zchain::load(Some(&empty), None, GEOM, 1.0) {
        Err(SidecarError::Empty) => {}
        other => panic!("expected the empty refusal, got {other:?}"),
    }

    // One gr file anywhere is a legal plugin shape on its own
    // (core_v41_amp.c:258-260).
    fs::write(empty.join("gr_L01.bin"), gr_f32(4, 2, &[1.0; 8])).unwrap();
    let z = V41Zchain::load(Some(&empty), None, GEOM, 1.0)
        .unwrap()
        .unwrap();
    assert_eq!(z.n_gr_layers(), 1);
    assert_eq!(z.n_rb_layers(), 0);
    assert_eq!(z.n_amp_layers(), 0);
}

#[test]
fn gr_merges_by_multiplication_across_both_dirs() {
    let z2 = tmpdir("gr-z2");
    let z3 = tmpdir("gr-z3");
    // Layer 0: both directories -> the element-wise product.
    fs::write(
        z2.join("gr_L00.bin"),
        gr_f32(4, 2, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    )
    .unwrap();
    fs::write(
        z3.join("gr_L00.bin"),
        gr_f32(4, 2, &[0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0]),
    )
    .unwrap();
    // Layer 1: ② only. Layer 2: ③ only.
    fs::write(z2.join("gr_L01.bin"), gr_f32(4, 2, &[1.5; 8])).unwrap();
    fs::write(z3.join("gr_L02.bin"), gr_f32(4, 2, &[0.25; 8])).unwrap();

    let z = V41Zchain::load(Some(&z2), Some(&z3), GEOM, 1.0)
        .unwrap()
        .unwrap();
    let g0 = z.gr[0].as_ref().unwrap();
    assert!(g0.from2 && g0.from3);
    assert_eq!(g0.factor, vec![1.0, 1.5, 2.0, 2.5, 12.0, 14.0, 16.0, 18.0]);
    let g1 = z.gr[1].as_ref().unwrap();
    assert!(g1.from2 && !g1.from3);
    assert_eq!(g1.factor, vec![1.5; 8]);
    let g2 = z.gr[2].as_ref().unwrap();
    assert!(!g2.from2 && g2.from3);
    assert_eq!(g2.factor, vec![0.25; 8]);
    assert!(z.gr[3].is_none());
    assert_eq!(z.n_gr_layers(), 3);
}

#[test]
fn gr_fp4_stores_s_minus_one_through_the_merge() {
    // One fp4 block of 32 elements: nibble 1 = 0.5, scale 1.0 -> factor 1.5.
    let mut blk = [0x11u8; 17];
    blk[16] = 127;
    let z2 = tmpdir("grfp4-z2");
    let z3 = tmpdir("grfp4-z3");
    fs::write(z2.join("gr_L00.bin"), gr_fp4(32, 1, &[blk])).unwrap();
    // ③ in f16: 1.0 (0x3C00) for all 32 elements.
    let mut f16_file = Vec::new();
    put_i32(&mut f16_file, 32);
    put_i32(&mut f16_file, 1);
    put_i32(&mut f16_file, 2);
    for _ in 0..32 {
        f16_file.extend_from_slice(&0x3C00u16.to_le_bytes());
    }
    fs::write(z3.join("gr_L00.bin"), &f16_file).unwrap();

    let geom = ZchainGeom {
        n_layer: 1,
        n_expert: 32,
        n_embd: 1,
    };
    let z = V41Zchain::load(Some(&z2), Some(&z3), geom, 1.0)
        .unwrap()
        .unwrap();
    let g = z.gr[0].as_ref().unwrap();
    assert!(g.factor.iter().all(|&v| v == 1.5), "1 + 0.5 times 1.0");
}

#[test]
fn rb_comes_from_the_zchain_directory_only() {
    let z2 = tmpdir("rb-z2");
    let z3 = tmpdir("rb-z3");
    fs::write(z2.join("rb_L01.bin"), rb(4, &[-1.0, -0.5, 0.5, 1.0])).unwrap();
    // ③ carries its own rb file; the engine never reads it (v41_rb_load is
    // called with the ② directory only, core_v41_amp.c:228).
    fs::write(z3.join("rb_L01.bin"), rb(4, &[9.0, 9.0, 9.0, 9.0])).unwrap();
    fs::write(z2.join("gr_L00.bin"), gr_f32(4, 2, &[1.0; 8])).unwrap();

    let z = V41Zchain::load(Some(&z2), Some(&z3), GEOM, 1.0)
        .unwrap()
        .unwrap();
    assert_eq!(z.rb[1].as_deref(), Some(&[-1.0, -0.5, 0.5, 1.0][..]));
    assert!(z.rb[0].is_none());
    assert_eq!(z.n_rb_layers(), 1);
}

#[test]
fn amp_concatenates_by_rank_with_beta_on_two_only() {
    let z2 = tmpdir("amp-z2");
    let z3 = tmpdir("amp-z3");
    // ② K=2: A rows [1,2],[3,4]; B rows [10,20],[30,40].
    fs::write(
        z2.join("amp_L00.bin"),
        amp_f32(2, 2, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]),
    )
    .unwrap();
    // ③ K=1: A row [5,6]; B row [50,60].
    fs::write(
        z3.join("amp_L00.bin"),
        amp_f32(2, 1, &[5.0, 6.0], &[50.0, 60.0]),
    )
    .unwrap();
    // Layer 1: ② only, layer 2: ③ only.
    fs::write(
        z2.join("amp_L01.bin"),
        amp_f32(2, 1, &[7.0, 8.0], &[70.0, 80.0]),
    )
    .unwrap();
    fs::write(
        z3.join("amp_L02.bin"),
        amp_f32(2, 2, &[1.0, 0.0, 0.0, 1.0], &[0.5, 0.5, 0.5, 0.5]),
    )
    .unwrap();

    let z = V41Zchain::load(Some(&z2), Some(&z3), GEOM, 0.5)
        .unwrap()
        .unwrap();
    let a0 = z.amp[0].as_ref().unwrap();
    assert_eq!((a0.k2, a0.k3, a0.k()), (2, 1, 3));
    assert_eq!((a0.typ2, a0.typ3), (1, 1));
    // ②'s A rows halved by beta, ③ untouched; B untouched on both.
    assert_eq!(a0.a, vec![0.5, 1.0, 1.5, 2.0, 5.0, 6.0]);
    assert_eq!(a0.b, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let a1 = z.amp[1].as_ref().unwrap();
    assert_eq!((a1.k2, a1.k3), (1, 0));
    assert_eq!(a1.a, vec![3.5, 4.0]);
    assert_eq!(a1.b, vec![70.0, 80.0]);
    let a2 = z.amp[2].as_ref().unwrap();
    assert_eq!((a2.k2, a2.k3), (0, 2));
    assert_eq!(a2.a, vec![1.0, 0.0, 0.0, 1.0]);
    assert!(z.amp[3].is_none());
    assert_eq!(z.n_amp_layers(), 3);
    assert_eq!(z.k_range(), Some((1, 3)));

    // A non-positive beta is clamped to 1.0 (core_v41_api.c:83).
    let z = V41Zchain::load(Some(&z2), Some(&z3), GEOM, 0.0)
        .unwrap()
        .unwrap();
    assert_eq!(z.beta, 1.0);
    assert_eq!(z.amp[0].as_ref().unwrap().a[0], 1.0);
}

#[test]
fn amp_rank_sum_over_the_cap_refuses() {
    let z2 = tmpdir("rank-z2");
    let z3 = tmpdir("rank-z3");
    fs::write(
        z2.join("amp_L00.bin"),
        amp_f32(2, 5000, &[0.0; 10000], &[0.0; 10000]),
    )
    .unwrap();
    fs::write(
        z3.join("amp_L00.bin"),
        amp_f32(2, 4000, &[0.0; 8000], &[0.0; 8000]),
    )
    .unwrap();
    match V41Zchain::load(Some(&z2), Some(&z3), GEOM, 1.0) {
        Err(SidecarError::RankOver {
            layer: 0,
            k2: 5000,
            k3: 4000,
        }) => {}
        other => panic!("expected the rank refusal, got {other:?}"),
    }
}

#[test]
fn a_broken_two_file_stops_before_the_three_is_read() {
    let z2 = tmpdir("stop-z2");
    let z3 = tmpdir("stop-z3");
    // ②'s amp is truncated; ③'s amp is fine. The engine reads ② first and
    // returns before touching ③ (core_v41_amp.c:238).
    let mut bad = amp_f32(2, 1, &[1.0, 2.0], &[3.0, 4.0]);
    bad.truncate(14);
    fs::write(z2.join("amp_L00.bin"), &bad).unwrap();
    fs::write(
        z3.join("amp_L00.bin"),
        amp_f32(2, 1, &[5.0, 6.0], &[7.0, 8.0]),
    )
    .unwrap();
    match V41Zchain::load(Some(&z2), Some(&z3), GEOM, 1.0) {
        Err(SidecarError::Truncated(p)) => assert!(p.contains("stop-z2"), "{p}"),
        other => panic!("expected the ② truncation, got {other:?}"),
    }

    // A broken gr file is a refusal too, not a silent skip.
    let z2 = tmpdir("stopgr-z2");
    let mut bad = gr_f32(4, 2, &[1.0; 8]);
    bad.truncate(13);
    fs::write(z2.join("gr_L00.bin"), &bad).unwrap();
    match V41Zchain::load(Some(&z2), None, GEOM, 1.0) {
        Err(SidecarError::Truncated(_)) => {}
        other => panic!("expected the gr truncation, got {other:?}"),
    }
}

#[test]
fn base_fnv_gate_runs_before_any_layer_is_read() {
    let z2 = tmpdir("gate-z2");
    let z3 = tmpdir("gate-z3");
    // The ② directory is garbage: a broken gr file. A mismatched base.fnv in
    // ③ must refuse first, so the broken file is never reached.
    let mut bad = gr_f32(4, 2, &[1.0; 8]);
    bad.truncate(13);
    fs::write(z2.join("gr_L00.bin"), &bad).unwrap();
    fs::write(z3.join("base.fnv"), "0000000000000001 1\n").unwrap();
    match V41Zchain::load(Some(&z2), Some(&z3), GEOM, 1.0) {
        Err(SidecarError::BaseMismatch {
            want: 1,
            want_files: 1,
            ..
        }) => {}
        other => panic!("expected the gate refusal, got {other:?}"),
    }

    // With the fingerprint matching, the same garbage is reached and refused
    // there instead.
    let (hash, files) = gr_dir_fnv(&z2, GEOM.n_layer);
    fs::write(z3.join("base.fnv"), format!("{hash:016x} {files}\n")).unwrap();
    match V41Zchain::load(Some(&z2), Some(&z3), GEOM, 1.0) {
        Err(SidecarError::Truncated(_)) => {}
        other => panic!("expected the gr truncation after the gate, got {other:?}"),
    }

    // A ③ directory without base.fnv is absent, which passes.
    let bare = tmpdir("gate-bare");
    fs::write(bare.join("gr_L00.bin"), gr_f32(4, 2, &[2.0; 8])).unwrap();
    let z = V41Zchain::load(None, Some(&bare), GEOM, 1.0)
        .unwrap()
        .unwrap();
    assert_eq!(z.base, BaseFingerprint::Absent);
    assert_eq!(z.gr[0].as_ref().unwrap().factor, vec![2.0; 8]);
}
