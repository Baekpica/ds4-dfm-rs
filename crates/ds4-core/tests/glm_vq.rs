//! GVQ1 containers use their own byte geometry, never a GGML block type.
use ds4_core::TensorInventory;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const VQ: u32 = 0x4756_5138;
const BOOK: u32 = 0x4756_4342;
const FP8: u32 = 0x4751_4638;
const ALIGN: usize = 128;
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ds4-gvq-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        Self(path)
    }
    fn open(&self, bytes: &[u8]) -> Result<TensorInventory, ds4_core::TensorError> {
        fs::write(&self.0, bytes).unwrap();
        TensorInventory::open(&self.0)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn align(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

struct Tensor<'a> {
    name: &'a str,
    dims: Vec<u64>,
    typ: u32,
    bytes: Vec<u8>,
}

// Independent fixture writer. Indices use a bit-at-a-time low plane, and
// the high bit deliberately alternates across row and byte boundaries.
fn fixture(k: usize, precision: usize) -> (Vec<u8>, usize) {
    let rows = 3;
    let cols = 128;
    let book = if precision == 1 {
        vec![0x38; k * 8]
    } else {
        [0x00, 0x3c].repeat(k * 8)
    };
    let mut payload = vec![0; 512];
    payload[0..4].copy_from_slice(b"GVQ1");
    payload[4..6].copy_from_slice(&1u16.to_le_bytes());
    payload[6] = if k == 4096 { 12 } else { 13 };
    payload[7] = 8;
    for (off, value) in [(8, rows), (12, cols), (16, k), (20, precision), (24, 12)] {
        payload[off..off + 4].copy_from_slice(&(value as u32).to_le_bytes());
    }
    let main = 256;
    let plane = if k == 8192 { 384 } else { 0 };
    let size = if k == 8192 { 512 } else { 384 };
    payload.truncate(size);
    for (off, value) in [(32, 128), (40, main), (48, plane), (56, size)] {
        payload[off..off + 8].copy_from_slice(&(value as u64).to_le_bytes());
    }
    for row in 0..rows {
        payload[128 + 2 * row..130 + 2 * row].copy_from_slice(&0x3800u16.to_le_bytes());
    }
    for j in 0..rows * cols / 8 {
        let index = (j * 1973) % k;
        for bit in 0..12 {
            payload[main + (12 * j + bit) / 8] |=
                (((index >> bit) & 1) << ((12 * j + bit) % 8)) as u8;
        }
        if plane != 0 {
            payload[plane + j / 8] |= (((index >> 12) & 1) << (j % 8)) as u8;
        }
    }
    payload[64..96].copy_from_slice(&Sha256::digest(&book));
    checksum(&mut payload);
    let tensors = [
        Tensor {
            name: "book",
            dims: vec![8, k as u64],
            typ: if precision == 1 { BOOK } else { 1 },
            bytes: book,
        },
        Tensor {
            name: "weight",
            dims: vec![cols as u64, rows as u64],
            typ: VQ,
            bytes: payload,
        },
    ];
    container(&[("glm_vq.codebooks", r#"{"12":"book"}"#)], &tensors)
}

fn checksum(payload: &mut [u8]) {
    let digest = Sha256::digest(&payload[128..]);
    payload[96..128].copy_from_slice(&digest);
}

fn container(meta: &[(&str, &str)], tensors: &[Tensor<'_>]) -> (Vec<u8>, usize) {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(3 + meta.len() as u64).to_le_bytes());
    string(&mut out, "general.architecture");
    out.extend_from_slice(&8u32.to_le_bytes());
    string(&mut out, "glm5_next");
    for (key, value) in [("general.alignment", 128u32), ("glm_vq.format_version", 1)] {
        string(&mut out, key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }
    for (key, value) in meta {
        string(&mut out, key);
        out.extend_from_slice(&8u32.to_le_bytes());
        string(&mut out, value);
    }
    let mut payload = Vec::new();
    for tensor in tensors {
        payload.resize(align(payload.len()), 0);
        string(&mut out, tensor.name);
        out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.typ.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        payload.extend_from_slice(&tensor.bytes);
    }
    let base = align(out.len());
    out.resize(base, 0);
    out.extend_from_slice(&payload);
    (out, base)
}

#[test]
fn private_geometry_is_validated() {
    let tmp = Temp::new();
    for k in [4096, 8192] {
        for precision in [1, 2] {
            let (bytes, _) = fixture(k, precision);
            let inv = tmp.open(&bytes).unwrap();
            assert_eq!(inv.find("book").unwrap().bytes, (k * 8 * precision) as u64);
            assert_eq!(
                inv.find("weight").unwrap().bytes,
                if k == 4096 { 384 } else { 512 }
            );
        }
    }
}

#[test]
fn malformed_payloads_fail_closed() {
    let tmp = Temp::new();
    let (bytes, base) = fixture(8192, 1);
    let start = base + 8192 * 8;
    for (off, value) in [
        (0, 0),
        (4, 2),
        (6, 12),
        (7, 4),
        (8, 4),
        (12, 64),
        (16, 0),
        (20, 0),
        (24, 16),
        (28, 1),
        (32, 64),
        (40, 128),
        (48, 0),
        (56, 0),
        (64, 0),
        (96, 0),
        (256, 255),
        (384, 255),
    ] {
        let mut corrupt = bytes.clone();
        corrupt[start + off] = if bytes[start + off] == value {
            value ^ 1
        } else {
            value
        };
        assert!(
            tmp.open(&corrupt).is_err(),
            "accepted corrupt header/payload at {off}"
        );
    }
    // Authentic checksums cannot legitimize NaN/zero gains or nonzero padding.
    for (off, value) in [
        (128, 0x7e00u16),
        (128, 0),
        (128, 0xbc00),
        (134, 1),
        (330, 1),
        (390, 1),
    ] {
        let mut corrupt = bytes.clone();
        corrupt[start + off..start + off + 2].copy_from_slice(&value.to_le_bytes());
        checksum(&mut corrupt[start..]);
        assert!(
            tmp.open(&corrupt).is_err(),
            "accepted invalid gain/padding at {off}"
        );
    }
    assert!(tmp.open(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(tmp.open(&trailing).is_err());
    let mut book_nan = bytes;
    book_nan[base] = 0x7f;
    let book_hash = Sha256::digest(&book_nan[base..start]);
    book_nan[start + 64..start + 96].copy_from_slice(&book_hash);
    assert!(tmp.open(&book_nan).is_err());
}

#[test]
fn native_fp8_scales() {
    let tmp = Temp::new();
    let mut tensors = [
        Tensor {
            name: "native.weight",
            dims: vec![128, 256],
            typ: FP8,
            bytes: vec![0x38; 128 * 256],
        },
        Tensor {
            name: "native.weight_scale_inv",
            dims: vec![1, 2],
            typ: 0,
            bytes: [0.5f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(),
        },
    ];
    let links = [(
        "glm_vq.fp8_scales",
        r#"{"native.weight":"native.weight_scale_inv"}"#,
    )];
    let (valid, _) = container(&links, &tensors);
    assert_eq!(
        tmp.open(&valid)
            .unwrap()
            .find("native.weight")
            .unwrap()
            .bytes,
        32768
    );
    assert!(tmp.open(&container(&[], &tensors).0).is_err());
    let wrong = [("glm_vq.fp8_scales", r#"{"native.weight":"native.weight"}"#)];
    assert!(tmp.open(&container(&wrong, &tensors).0).is_err());
    for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        tensors[1].bytes[..4].copy_from_slice(&value.to_le_bytes());
        assert!(tmp.open(&container(&links, &tensors).0).is_err());
    }
}

#[test]
#[ignore = "requires the pinned local handoff; no model download"]
fn handoff_parts_are_valid() {
    let root = PathBuf::from(std::env::var_os("DS4_GLM_VQ_FIXTURES").expect("DS4_GLM_VQ_FIXTURES"));
    let mut count = 0;
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path().join("fixture.gguf");
        if !path.is_file() {
            continue;
        }
        let inv = TensorInventory::open(&path).unwrap();
        assert!(inv.tensors.iter().all(|t| t.bytes > 0));
        count += 1;
    }
    assert_eq!(count, 10);
}

#[test]
fn reject_container_contracts() {
    let tmp = Temp::new();
    let mut tensors = vec![Tensor {
        name: "norm",
        dims: vec![2],
        typ: 1,
        bytes: vec![0; 4],
    }];
    let (mut bytes, base) = container(&[], &tensors);
    assert!(tmp.open(&bytes).is_ok());
    bytes[base - 1] = 1;
    assert!(tmp.open(&bytes).is_err());
    for typ in [42, 44, 142, VQ + 1] {
        tensors[0].typ = typ;
        assert!(tmp.open(&container(&[], &tensors).0).is_err());
    }
    tensors[0].typ = 1;
    tensors[0].dims = vec![0];
    assert!(tmp.open(&container(&[], &tensors).0).is_err());
    tensors[0].dims = vec![2];
    tensors.push(Tensor {
        name: "norm",
        dims: vec![2],
        typ: 1,
        bytes: vec![0; 4],
    });
    assert!(tmp.open(&container(&[], &tensors).0).is_err());
    tensors.pop();
    for meta in [
        vec![("general.architecture", "glm5_next")],
        vec![("glm_vq.codebooks", r#"{"1":"norm"}"#)],
        vec![("glm_vq.fp8_scales", r#"{"norm":"norm"}"#)],
        vec![("glm_vq.logical_shapes", r#"{"norm":[2,1,1,1,1]}"#)],
        vec![("glm_vq.codebooks", "[]")],
    ] {
        assert!(tmp.open(&container(&meta, &tensors).0).is_err());
    }
}
