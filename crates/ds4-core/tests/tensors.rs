//! C↔Rust GGUF tensor directory, nbytes, and split sibling remap. Synthetic only.

use ds4_core::{
    dump_apply_tapes, dump_consume_tapes, dump_nbytes_table, dump_sibling_script,
    model_split_sibling_path, TensorInventory,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_TENSOR_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/tensor_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/tensor_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_out(args: &[&str]) -> String {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run tensor_c_oracle");
    assert!(
        out.status.success(),
        "oracle {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
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

struct Tens<'a> {
    name: &'a str,
    dims: &'a [u64],
    typ: u32,
    rel: u64,
}

fn write_gguf(path: &Path, kvs: &[(&str, u32)], tensors: &[Tens<'_>]) {
    let mut buf = Vec::new();
    put_u32(&mut buf, 0x4655_4747);
    put_u32(&mut buf, 3);
    put_u64(&mut buf, tensors.len() as u64);
    put_u64(&mut buf, kvs.len() as u64);
    for (k, v) in kvs {
        put_str(&mut buf, k);
        put_u32(&mut buf, 4); // UINT32
        put_u32(&mut buf, *v);
    }
    for t in tensors {
        put_str(&mut buf, t.name);
        put_u32(&mut buf, t.dims.len() as u32);
        for d in t.dims {
            put_u64(&mut buf, *d);
        }
        put_u32(&mut buf, t.typ);
        put_u64(&mut buf, t.rel);
    }
    let align = 32u64;
    let data_pos = {
        let rem = (buf.len() as u64) % align;
        if rem == 0 {
            buf.len()
        } else {
            buf.len() + (align - rem) as usize
        }
    };
    buf.resize(data_pos, 0);
    let mut need = data_pos;
    for t in tensors {
        let elems: u64 = t.dims.iter().product();
        let bytes = ds4_core::tensor_nbytes(t.typ, elems).unwrap_or(0);
        let end = data_pos as u64 + t.rel + bytes;
        if end as usize > need {
            need = end as usize;
        }
    }
    buf.resize(need, 0);
    fs::write(path, buf).unwrap();
}

#[test]
fn nbytes_table_matches_c() {
    let rust = dump_nbytes_table();
    let c = c_out(&["nbytes"]);
    assert_eq!(rust, c);
}

#[test]
fn sibling_paths_match_c() {
    let cases = [
        ("/models/foo-00001-of-00003.gguf", 0u32, 3u32),
        ("/models/foo-00001-of-00003.gguf", 1, 3),
        ("/models/foo-00001-of-00003.gguf", 2, 3),
        ("/models/Motif-3-00001-of-00011.gguf", 10, 11),
        ("/nope.gguf", 0, 2),
        ("/models/foo-00002-of-00003.gguf", 0, 3),
        ("foo-00001-of-00003.gguf", 1, 3),
    ];
    for (path, index, count) in cases {
        let rust = dump_sibling_script(path, index, count);
        let c = c_out(&["sibling", path, &index.to_string(), &count.to_string()]);
        assert_eq!(rust, c, "sibling {path} {index} {count}");
    }
    assert_eq!(
        model_split_sibling_path("/m/a-00001-of-00003.gguf", 1, 3).as_deref(),
        Some("/m/a-00002-of-00003.gguf")
    );
}

#[test]
fn tensor_parse_matches_c() {
    let dir = std::env::temp_dir().join(format!("ds4-tensor-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.gguf");
    write_gguf(
        &path,
        &[("general.alignment", 32)],
        &[
            Tens {
                name: "token_embd.weight",
                dims: &[4, 8],
                typ: 8,
                rel: 0,
            },
            Tens {
                name: "blk.0.attn_norm.weight",
                dims: &[8],
                typ: 0,
                rel: 64,
            },
        ],
    );
    let rust = TensorInventory::open(&path).unwrap().dump();
    let c = c_out(&["parse", path.to_str().unwrap()]);
    if rust != c {
        panic!("parse mismatch\n--- rust ---\n{rust}\n--- c ---\n{c}");
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn split_inventory_concatenates_shards() {
    let dir = std::env::temp_dir().join(format!("ds4-split-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let a = dir.join("toy-00001-of-00002.gguf");
    let b = dir.join("toy-00002-of-00002.gguf");
    write_gguf(
        &a,
        &[("split.count", 2), ("split.tensors.count", 2)],
        &[Tens {
            name: "a.weight",
            dims: &[4],
            typ: 0,
            rel: 0,
        }],
    );
    write_gguf(
        &b,
        &[],
        &[Tens {
            name: "b.weight",
            dims: &[4],
            typ: 0,
            rel: 0,
        }],
    );
    let inv = TensorInventory::open(&a).unwrap();
    assert_eq!(inv.shards.len(), 2);
    assert_eq!(inv.tensors.len(), 2);
    assert_eq!(inv.tensors[0].name, "a.weight");
    assert_eq!(inv.tensors[1].name, "b.weight");
    assert_eq!(inv.tensors[0].shard, 0);
    assert_eq!(inv.tensors[1].shard, 1);
    assert_eq!(
        inv.tensors[1].abs_offset,
        inv.shards[1].base + {
            let g = ds4_core::GgufFile::open(&b).unwrap();
            // second tensor abs in its own file, then + base
            TensorInventory::from_file(&b, &g).unwrap().tensors[0].abs_offset
        }
    );
    assert!(inv.shards[1].base >= inv.shards[0].size);
    let _ = fs::remove_dir_all(&dir);
}

fn load_oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_LOAD_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/load_c_oracle")
}

#[test]
fn host_tensor_consume_matches_c() {
    let p = load_oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/load_c_oracle (missing {})",
        p.display()
    );
    let out = Command::new(&p)
        .args(["consume-tapes"])
        .output()
        .expect("run load_c_oracle");
    assert!(
        out.status.success(),
        "load_c_oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        dump_consume_tapes(),
        String::from_utf8(out.stdout).expect("oracle utf8")
    );
}

#[test]
fn host_tensor_apply_matches_c() {
    let p = load_oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/load_c_oracle (missing {})",
        p.display()
    );
    let out = Command::new(&p)
        .args(["apply-tapes"])
        .output()
        .expect("run load_c_oracle");
    assert!(
        out.status.success(),
        "load_c_oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        dump_apply_tapes(),
        String::from_utf8(out.stdout).expect("oracle utf8")
    );
}
