//! VQ blob decode tests.
//!
//! Fixtures are built byte by byte from the format the C engine defines
//! (`/data/YoungAi` commit 3946dbc): `vq_fmt.h` for the container and the v2
//! payload, `src/cuda/cuda_vq_row.inc.cu` for the v3 geometry. The expected
//! values are computed from the fixture's own codebook, indices and gains, so
//! a parsing slip shows up as a value mismatch rather than a silent pass.

use ds4_core::{blob_ok, e4m3fn_to_f32, f16_to_f32, VqError, VqMatrix};

const BLOB_MAGIC: u32 = 0x4C56_5144; // 'DQVL' (vq_fmt.h:15)
const MAT_MAGIC: u32 = 0x5156_5144; // 'DQVQ' v2 payload (vq_fmt.h:16)
const MAT3_MAGIC: u32 = 0x3356_5144; // 'DQV3' v3 payload (vq_fmt.h:19)
const V3_FLAG_E4M3: u32 = 1; // cuda_vq_row.inc.cu:35
const V3_FLAG_PLANE: u32 = 2; // cuda_vq_row.inc.cu:41

/// f32 -> f16 for values that are exact in half precision; asserts exactness.
fn f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xFF) as i32 - 127 + 15) as u16;
    let man = ((bits >> 13) & 0x3FF) as u16;
    assert!(exp > 0 && exp < 31, "value not a normal half");
    let h = sign | (exp << 10) | man;
    assert_eq!(f16_to_f32(h), f, "value not exact in half");
    h
}

fn push_u16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// `[DQVL][ver][L][nexp][nexp*3 u64 slot table]` (vq_fmt.h:2-4).
fn blob_header(ver: u32, nexp: u32, slots: &[u64]) -> Vec<u8> {
    let mut b = Vec::new();
    push_u32(&mut b, BLOB_MAGIC);
    push_u32(&mut b, ver);
    push_u32(&mut b, 0);
    push_u32(&mut b, nexp);
    for s in slots {
        push_u64(&mut b, *s);
    }
    b
}

/// v2 payload: `[DQVQ][dim][nc][rows][cols][nc*dim f16][rows f16][bitstream]`.
fn v2_payload(nc: u16, rows: u32, cols: u32, stream: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    push_u32(&mut b, MAT_MAGIC);
    push_u16(&mut b, 8);
    push_u16(&mut b, nc);
    push_u32(&mut b, rows);
    push_u32(&mut b, cols);
    for w in 0..u32::from(nc) {
        for d in 0..8u32 {
            push_u16(&mut b, f16((w + 1) as f32 * (d + 1) as f32));
        }
    }
    for r in 0..rows {
        push_u16(&mut b, f16(if r == 0 { 0.5 } else { 2.0 }));
    }
    b.extend_from_slice(stream);
    b
}

#[test]
fn v2_decode_matches_the_format() {
    // rows 2, cols 16 -> two 8-wide groups per row; nc 4 -> 2-bit indices.
    // Row 0 picks [3, 1], row 1 picks [0, 2] (vq_fmt.h:75-80 packs them
    // little-endian across the whole stream, not per row).
    let stream = [0x87u8, 0x00, 0x00]; // 3|1<<2|0<<4|2<<6, plus the safety bytes
    let payload = v2_payload(4, 2, 16, &stream);
    let mut blob = blob_header(2, 1, &[40, 0, 0]);
    assert_eq!(blob.len(), 40);
    blob.extend_from_slice(&payload);

    assert!(blob_ok(&blob));
    let m = VqMatrix::open(&blob, 0, 0, 2, 16).expect("open");
    let got = m.dequant_f32(None);

    let cb = |w: u32, d: u32| (w + 1) as f32 * (d + 1) as f32;
    let picks = [[3u32, 1u32], [0u32, 2u32]];
    let gains = [0.5f32, 2.0f32];
    for r in 0..2usize {
        for i in 0..2usize {
            for d in 0..8usize {
                let want = cb(picks[r][i], d as u32) * gains[r];
                assert_eq!(got[r * 16 + i * 8 + d], want, "row {r} group {i} dim {d}");
            }
        }
    }
}

#[test]
fn v2_absent_slot_is_no_matrix() {
    let blob = blob_header(2, 1, &[0, 0, 0]);
    assert!(matches!(
        VqMatrix::open(&blob, 0, 0, 2, 16),
        Err(VqError::NoMatrix)
    ));
}

fn v3_payload(nc: u16, rows: u32, cols: u32, flags: u32, stream: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    push_u32(&mut b, MAT3_MAGIC);
    push_u16(&mut b, 8);
    push_u16(&mut b, nc);
    push_u32(&mut b, rows);
    push_u32(&mut b, cols);
    push_u32(&mut b, flags);
    push_u32(&mut b, 12); // mnb: the main stream is always 12 bits
    push_u64(&mut b, 0); // cb_off, patched by the caller
    for _ in 0..rows {
        push_u16(&mut b, f16(1.0));
    }
    b.extend_from_slice(stream);
    b
}

/// Codebook byte pattern: 1.0, 1.125, 1.25, 1.375 cycling by index.
fn cb_byte(idx: usize) -> u8 {
    0x38 + (idx % 4) as u8
}

/// A complete v3 blob: header, payload with `cb_off` pointing past it, then
/// the layer codebook the payload shares (cuda_vq_row.inc.cu:33-40).
fn v3_blob(nc: u16, rows: u32, cols: u32, flags: u32, stream: &[u8]) -> Vec<u8> {
    let payload = v3_payload(nc, rows, cols, flags, stream);
    let mut blob = blob_header(3, 1, &[40, 0, 0]);
    assert_eq!(blob.len(), 40);
    blob.extend_from_slice(&payload);
    let cb_off = blob.len() as u64;
    for idx in 0..usize::from(nc) {
        for _ in 0..8 {
            blob.push(cb_byte(idx));
        }
    }
    blob[40 + 24..40 + 32].copy_from_slice(&cb_off.to_le_bytes());
    blob
}

#[test]
fn v3_12bit_decode_matches_the_format() {
    // rows 1, cols 16 -> two indices; nc 4096 -> 12-bit main stream, no plane.
    let blob = v3_blob(4096, 1, 16, V3_FLAG_E4M3, &[0x05, 0xE0, 0xFF, 0x00]);

    assert!(blob_ok(&blob));
    let m = VqMatrix::open(&blob, 0, 0, 1, 16).expect("open");
    assert_eq!(m.nbit, 12);
    let got = m.dequant_f32(None);
    for (i, idx) in [5usize, 4094].iter().enumerate() {
        for d in 0..8usize {
            let want = e4m3fn_to_f32(cb_byte(*idx));
            assert_eq!(got[i * 8 + d], want, "group {i} dim {d}");
        }
    }
}

#[test]
fn v3_13bit_plane_decode_matches_the_format() {
    // nc 8192 -> 13 bits: the 12-bit main stream carries the low bits and the
    // plane bit j is the 13th bit of index j (cuda_vq_row.inc.cu:199-203).
    let blob = v3_blob(
        8192,
        1,
        16,
        V3_FLAG_E4M3 | V3_FLAG_PLANE,
        &[0x05, 0xE0, 0xFF, 0x00, 0x03],
    );

    assert!(blob_ok(&blob));
    let m = VqMatrix::open(&blob, 0, 0, 1, 16).expect("open");
    assert_eq!(m.nbit, 13);
    let got = m.dequant_f32(None);
    for (i, idx) in [0x1005usize, 0x1FFE].iter().enumerate() {
        for d in 0..8usize {
            let want = e4m3fn_to_f32(cb_byte(*idx));
            assert_eq!(got[i * 8 + d], want, "group {i} dim {d}");
        }
    }
}

#[test]
fn gain_override_replaces_the_payload_gain() {
    let blob = v3_blob(4096, 1, 16, V3_FLAG_E4M3, &[0x00, 0x00, 0x00, 0x00]);

    let m = VqMatrix::open(&blob, 0, 0, 1, 16).expect("open");
    let gov = [4.0f32];
    let got = m.dequant_f32(Some(&gov));
    let want = e4m3fn_to_f32(cb_byte(0)) * 4.0;
    assert_eq!(got[0], want);
}

#[test]
fn v2_matches_the_engine_decoder() {
    // The engine's own host decoder is the authority for the v2 payload. The
    // fixtures and the reference values are produced by
    // tests/fixtures/vq/gen_v2_ref.c, which includes the vendored vq_fmt.h
    // (copied verbatim from /data/YoungAi at 3946dbc) and calls
    // ds4vq_dequant_f32. Regenerate with:
    //   cc -O2 -D__fp16=float -I. gen_v2_ref.c -o gen_v2_ref && ./gen_v2_ref
    // -D__fp16=float neutralizes the f16-output variant, which this harness
    // never calls.
    let cases: [(&[u8], &[u8]); 2] = [
        (
            include_bytes!("fixtures/vq/v2_nc256.blob"),
            include_bytes!("fixtures/vq/v2_nc256.ref.f32"),
        ),
        (
            include_bytes!("fixtures/vq/v2_nc4096.blob"),
            include_bytes!("fixtures/vq/v2_nc4096.ref.f32"),
        ),
    ];
    for (blob, rref) in cases {
        let m = VqMatrix::open(blob, 0, 0, 16, 64).expect("open");
        let got = m.dequant_f32(None);
        assert_eq!(got.len() * 4, rref.len());
        for (i, chunk) in rref.chunks_exact(4).enumerate() {
            let want = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert_eq!(got[i].to_bits(), want.to_bits(), "value {i}");
        }
    }
}

#[test]
fn blob_ok_rejects_bad_headers() {
    let good = blob_header(3, 1, &[40, 0, 0]);
    assert!(blob_ok(&good));

    let mut bad_magic = good.clone();
    bad_magic[0] = b'X';
    assert!(!blob_ok(&bad_magic));

    let bad_version = blob_header(4, 1, &[40, 0, 0]);
    assert!(!blob_ok(&bad_version));

    let bad_nexp = blob_header(3, 0, &[]);
    assert!(!blob_ok(&bad_nexp));

    assert!(!blob_ok(&good[..15]));
}

#[test]
fn open_rejects_shape_and_dim_mismatch() {
    let payload = v2_payload(4, 2, 16, &[0x87, 0x00, 0x00]);
    let mut blob = blob_header(2, 1, &[40, 0, 0]);
    blob.extend_from_slice(&payload);
    assert!(matches!(
        VqMatrix::open(&blob, 0, 0, 4, 16),
        Err(VqError::BadShape)
    ));

    let mut wrong_dim = blob.clone();
    wrong_dim[40 + 4] = 4; // dim = 4 instead of 8
    assert!(matches!(
        VqMatrix::open(&wrong_dim, 0, 0, 2, 16),
        Err(VqError::BadDim(4))
    ));
}
