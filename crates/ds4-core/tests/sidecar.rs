//! V4.1 sidecar gates, model-free: the fp4x32 block decoder against the
//! engine's own function (fixtures/sidecar/gen_fp4_ref.c), the three sidecar
//! readers against synthetic files, and the `base.fnv` admission rule.
//!
//! The live check - the posttrain directory's `base.fnv` against the ②
//! directories it could name - runs the inspector on the Spark; the value the
//! solver wrote is 0xf4a3fd3988135e75 over 66 files, and the reader must land
//! on it (port plan §6.6.3).

use std::fs;
use std::path::PathBuf;

use ds4_core::{
    check_base_fnv, deq_fp4x32, gr_dir_fnv, AmpSidecar, BaseFingerprint, GrSidecar, RbSidecar,
    SidecarError,
};

fn tmp(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join("ds4-sidecar");
    fs::create_dir_all(&dir).unwrap();
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    dir.join(format!("{n}-{name}"))
}

#[test]
fn fp4_decode_matches_the_engine_bit_for_bit() {
    let blocks = include_bytes!("fixtures/sidecar/fp4_blocks.bin");
    let want = include_bytes!("fixtures/sidecar/fp4_ref.f32");
    let got = deq_fp4x32(blocks);
    assert_eq!(got.len() * 4, want.len());
    for (i, (g, w)) in got.iter().zip(want.chunks_exact(4)).enumerate() {
        let w = f32::from_le_bytes(w.try_into().unwrap());
        assert_eq!(g.to_bits(), w.to_bits(), "value {i}");
    }
}

fn put_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[test]
fn gr_reads_all_three_storage_types() {
    // f32: factors are the stored values.
    let mut f32_file = Vec::new();
    put_i32(&mut f32_file, 4);
    put_i32(&mut f32_file, 3);
    put_i32(&mut f32_file, 1);
    for t in 0..12 {
        f32_file.extend_from_slice(&(1.0 + t as f32 * 0.5).to_le_bytes());
    }
    let p = tmp("gr_f32.bin");
    fs::write(&p, &f32_file).unwrap();
    let gr = GrSidecar::read(&p, 4, 3).unwrap().unwrap();
    assert_eq!(gr.typ, 1);
    assert_eq!(gr.factor[0], 1.0);
    assert_eq!(gr.factor[11], 1.0 + 11.0 * 0.5);

    // f16: 1.0 and 1.5 in half precision.
    let mut f16_file = Vec::new();
    put_i32(&mut f16_file, 4);
    put_i32(&mut f16_file, 3);
    put_i32(&mut f16_file, 2);
    for t in 0..12 {
        let v: u16 = if t % 2 == 0 { 0x3C00 } else { 0x3E00 };
        f16_file.extend_from_slice(&v.to_le_bytes());
    }
    let p = tmp("gr_f16.bin");
    fs::write(&p, &f16_file).unwrap();
    let gr = GrSidecar::read(&p, 4, 3).unwrap().unwrap();
    assert_eq!(gr.typ, 2);
    assert_eq!(gr.factor[0], 1.0);
    assert_eq!(gr.factor[1], 1.5);

    // fp4x32: the disk stores s-1, the reader restores 1 + raw. One block of
    // 32 elements covers 12 with nel=12? No: nel must be a multiple of 32, so
    // use 32 experts x 1 channel.
    let mut fp4_file = Vec::new();
    put_i32(&mut fp4_file, 32);
    put_i32(&mut fp4_file, 1);
    put_i32(&mut fp4_file, 43);
    let mut blk = [0u8; 17];
    for j in 0..16 {
        blk[j] = 0x11; // nibble 1 both halves = 0.5 * 2^0
    }
    blk[16] = 127; // scale 1.0
    fp4_file.extend_from_slice(&blk);
    let p = tmp("gr_fp4.bin");
    fs::write(&p, &fp4_file).unwrap();
    let gr = GrSidecar::read(&p, 32, 1).unwrap().unwrap();
    assert_eq!(gr.typ, 43);
    assert!(gr.factor.iter().all(|&v| v == 1.5), "0.5 restored to 1.5");

    // No file is not an error; a wrong header is.
    assert!(GrSidecar::read(&tmp("missing.bin"), 4, 3)
        .unwrap()
        .is_none());
    let mut bad = f32_file.clone();
    bad[0] = 5;
    let p = tmp("gr_bad.bin");
    fs::write(&p, &bad).unwrap();
    assert!(matches!(
        GrSidecar::read(&p, 4, 3),
        Err(SidecarError::Header(_))
    ));
}

#[test]
fn rb_reads_and_refuses() {
    let mut file = Vec::new();
    put_i32(&mut file, 4);
    put_i32(&mut file, 1);
    for t in 0..4 {
        file.extend_from_slice(&(t as f32 - 1.5).to_le_bytes());
    }
    let p = tmp("rb.bin");
    fs::write(&p, &file).unwrap();
    let rb = RbSidecar::read(&p, 4).unwrap().unwrap();
    assert_eq!(rb.bias, vec![-1.5, -0.5, 0.5, 1.5]);
    let mut bad = file.clone();
    bad[4] = 2; // type must be 1
    let p = tmp("rb_bad.bin");
    fs::write(&p, &bad).unwrap();
    assert!(matches!(
        RbSidecar::read(&p, 4),
        Err(SidecarError::Header(_))
    ));
    let p = tmp("rb_short.bin");
    fs::write(&p, &file[..12]).unwrap();
    assert!(matches!(
        RbSidecar::read(&p, 4),
        Err(SidecarError::Truncated(_))
    ));
}

#[test]
fn amp_reads_both_storage_types_and_refuses_bad_headers() {
    // f32: D=2, K=3, A then B.
    let mut file = Vec::new();
    put_i32(&mut file, 2);
    put_i32(&mut file, 3);
    put_i32(&mut file, 1);
    for t in 0..6 {
        file.extend_from_slice(&(t as f32).to_le_bytes());
    }
    for t in 0..6 {
        file.extend_from_slice(&(100.0 + t as f32).to_le_bytes());
    }
    let p = tmp("amp_f32.bin");
    fs::write(&p, &file).unwrap();
    let amp = AmpSidecar::read(&p, 2).unwrap().unwrap();
    assert_eq!(amp.k, 3);
    assert_eq!(amp.a[5], 5.0);
    assert_eq!(amp.b[0], 100.0);

    // fp4x32: K*D must be a multiple of 32.
    let mut file = Vec::new();
    put_i32(&mut file, 1);
    put_i32(&mut file, 32);
    put_i32(&mut file, 43);
    let blk = [0x77u8; 16];
    file.extend_from_slice(&blk);
    file.push(127);
    file.extend_from_slice(&blk);
    file.push(127);
    let p = tmp("amp_fp4.bin");
    fs::write(&p, &file).unwrap();
    let amp = AmpSidecar::read(&p, 1).unwrap().unwrap();
    assert_eq!(amp.typ, 43);
    assert!(amp.a.iter().all(|&v| v == 6.0));
    assert!(amp.b.iter().all(|&v| v == 6.0));

    // D must match the embedding width; K is bounded; the type is 1 or 43.
    let mut bad = file.clone();
    bad[0] = 3;
    let p = tmp("amp_bad_d.bin");
    fs::write(&p, &bad).unwrap();
    assert!(matches!(
        AmpSidecar::read(&p, 1),
        Err(SidecarError::Header(_))
    ));
    let mut bad = file.clone();
    bad[4..8].copy_from_slice(&9000i32.to_le_bytes());
    let p = tmp("amp_bad_k.bin");
    fs::write(&p, &bad).unwrap();
    assert!(matches!(
        AmpSidecar::read(&p, 1),
        Err(SidecarError::Header(_))
    ));
    let mut bad = file.clone();
    bad[8..12].copy_from_slice(&2i32.to_le_bytes());
    let p = tmp("amp_bad_ty.bin");
    fs::write(&p, &bad).unwrap();
    assert!(matches!(
        AmpSidecar::read(&p, 1),
        Err(SidecarError::Type(_))
    ));
}

#[test]
fn base_fnv_gate_matches_mutates_and_absents() {
    let dir = tmp("ampdir");
    fs::create_dir_all(&dir).unwrap();
    // A directory with gr/rb files on layers 0 and 2 (layer 1 missing).
    fs::write(dir.join("gr_L00.bin"), [1u8, 2, 3]).unwrap();
    fs::write(dir.join("rb_L00.bin"), [4u8, 5]).unwrap();
    fs::write(dir.join("gr_L02.bin"), [6u8]).unwrap();
    let (hash, files) = gr_dir_fnv(&dir, 40);
    assert_eq!(files, 3);

    let pt = tmp("ptdir");
    fs::create_dir_all(&pt).unwrap();
    fs::write(pt.join("gr_L39.bin"), [9u8]).unwrap();
    fs::write(pt.join("base.fnv"), format!("{hash:016x} {files}\n")).unwrap();
    assert_eq!(
        check_base_fnv(&pt, Some(&dir), 40).unwrap(),
        BaseFingerprint::Checked { hash, files }
    );

    // One byte changes in the ② directory: the ③ corrections no longer match.
    fs::write(dir.join("gr_L02.bin"), [7u8]).unwrap();
    match check_base_fnv(&pt, Some(&dir), 40) {
        Err(SidecarError::BaseMismatch {
            want,
            want_files,
            have_files,
            ..
        }) => {
            assert_eq!(want, hash);
            assert_eq!(want_files, 3);
            assert_eq!(have_files, 3);
        }
        other => panic!("expected a mismatch, got {other:?}"),
    }

    // No base.fnv at all: absent, which the caller reports and passes.
    let bare = tmp("bare");
    fs::create_dir_all(&bare).unwrap();
    assert_eq!(
        check_base_fnv(&bare, Some(&dir), 40).unwrap(),
        BaseFingerprint::Absent
    );

    // Without a ② directory the expected fingerprint is the seed and zero
    // files (core_v41_amp.c:167-168).
    let seed_only = tmp("seed");
    fs::create_dir_all(&seed_only).unwrap();
    fs::write(
        seed_only.join("base.fnv"),
        format!("{:016x} 0", ds4_core::FNV_SEED),
    )
    .unwrap();
    assert!(matches!(
        check_base_fnv(&seed_only, None, 40),
        Ok(BaseFingerprint::Checked { files: 0, .. })
    ));
}
