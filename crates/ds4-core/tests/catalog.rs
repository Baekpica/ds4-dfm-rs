//! mmap GGUF v3 identify vs C catalog oracle. Synthetic metadata only.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_CATALOG_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/catalog_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/catalog_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_parse(path: &Path) -> String {
    let out = Command::new(require_oracle())
        .arg(path)
        .output()
        .expect("run catalog_c_oracle");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("oracle utf8")
}

enum Val<'a> {
    U16(u16),
    U32(u32),
    I32(i32),
    Str(&'a str),
    ArrayU32(&'a [u32]),
    ArrayStr(&'a [&'a str]),
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
            Val::U16(v) => {
                put_u32(&mut buf, 2);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Val::U32(v) => {
                put_u32(&mut buf, 4);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Val::I32(v) => {
                put_u32(&mut buf, 5);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            Val::Str(s) => {
                put_u32(&mut buf, 8);
                put_str(&mut buf, s);
            }
            Val::ArrayU32(items) => {
                put_u32(&mut buf, 9);
                put_u32(&mut buf, 4);
                put_u64(&mut buf, items.len() as u64);
                for x in *items {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            }
            Val::ArrayStr(items) => {
                put_u32(&mut buf, 9);
                put_u32(&mut buf, 8);
                put_u64(&mut buf, items.len() as u64);
                for s in *items {
                    put_str(&mut buf, s);
                }
            }
        }
    }
    while buf.len() < 32 {
        buf.push(0);
    }
    fs::write(path, buf).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ds4-catalog-parity");
    fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn assert_same(path: &Path) {
    let c = c_parse(path);
    let rust = ds4_core::dump_parse(path);
    assert_eq!(rust, c, "mismatch for {}", path.display());
}

const FLASH: &[(&str, u32)] = &[
    ("deepseek4.block_count", 43),
    ("deepseek4.embedding_length", 4096),
    ("deepseek4.vocab_size", 129280),
    ("deepseek4.attention.head_count", 64),
    ("deepseek4.attention.head_count_kv", 1),
    ("deepseek4.attention.key_length", 512),
    ("deepseek4.attention.value_length", 512),
    ("deepseek4.rope.dimension_count", 64),
    ("deepseek4.attention.q_lora_rank", 1024),
    ("deepseek4.attention.output_lora_rank", 1024),
    ("deepseek4.attention.output_group_count", 8),
    ("deepseek4.expert_count", 256),
    ("deepseek4.expert_used_count", 6),
    ("deepseek4.expert_feed_forward_length", 2048),
    ("deepseek4.expert_shared_count", 1),
    ("deepseek4.hash_layer_count", 3),
    ("deepseek4.attention.sliding_window", 128),
    ("deepseek4.attention.indexer.head_count", 64),
    ("deepseek4.attention.indexer.key_length", 128),
    ("deepseek4.attention.indexer.top_k", 512),
    ("deepseek4.hyper_connection.count", 4),
    ("deepseek4.hyper_connection.sinkhorn_iterations", 20),
];

fn ds_kvs<'a>(arch: Option<&'a str>, dims: &'a [(&'a str, u32)]) -> Vec<(&'a str, Val<'a>)> {
    let mut v = Vec::new();
    if let Some(a) = arch {
        v.push(("general.architecture", Val::Str(a)));
    }
    for (k, n) in dims {
        v.push((*k, Val::U32(*n)));
    }
    v
}

#[test]
fn identify_families_and_errors_match_c() {
    let flash = tmp("flash.gguf");
    write_gguf(&flash, &ds_kvs(None, FLASH));
    assert_same(&flash);

    let pro_dims: Vec<(&str, u32)> = FLASH
        .iter()
        .map(|(k, v)| {
            (
                *k,
                match *k {
                    "deepseek4.block_count" => 61,
                    "deepseek4.embedding_length" => 7168,
                    "deepseek4.attention.head_count" => 128,
                    "deepseek4.attention.q_lora_rank" => 1536,
                    "deepseek4.attention.output_group_count" => 16,
                    "deepseek4.expert_count" => 384,
                    "deepseek4.expert_feed_forward_length" => 3072,
                    "deepseek4.attention.indexer.top_k" => 1024,
                    _ => *v,
                },
            )
        })
        .collect();
    let pro = tmp("pro.gguf");
    write_gguf(&pro, &ds_kvs(Some("deepseek4"), &pro_dims));
    assert_same(&pro);

    let miss = tmp("miss.gguf");
    let mut bad = ds_kvs(Some("deepseek4"), FLASH);
    if let Some((_, Val::U32(v))) = bad.iter_mut().find(|(k, _)| *k == "deepseek4.block_count") {
        *v = 1;
    }
    write_gguf(&miss, &bad);
    assert_same(&miss);

    let motif = tmp("motif.gguf");
    write_gguf(&motif, &[("general.architecture", Val::Str("motif3"))]);
    assert_same(&motif);

    let solar = tmp("solar.gguf");
    write_gguf(&solar, &[("general.architecture", Val::Str("solar-open2"))]);
    assert_same(&solar);

    let exa = tmp("exaone.gguf");
    write_gguf(&exa, &[("general.architecture", Val::Str("exaone-moe"))]);
    assert_same(&exa);

    let dots = tmp("dots3.gguf");
    write_gguf(&dots, &[("general.architecture", Val::Str("dots3-note"))]);
    assert_same(&dots);

    let qwen = tmp("qwen.gguf");
    write_gguf(&qwen, &[("general.architecture", Val::Str("qwen4exp"))]);
    assert_same(&qwen);

    let glm = tmp("glm.gguf");
    write_gguf(&glm, &[("general.architecture", Val::Str("glm-dsa"))]);
    assert_same(&glm);

    let split = tmp("split.gguf");
    write_gguf(
        &split,
        &[
            ("general.architecture", Val::Str("motif3")),
            ("split.count", Val::U16(3)),
            ("general.alignment", Val::U32(64)),
            ("tags", Val::ArrayStr(&["a", "b"])),
            ("nums", Val::ArrayU32(&[1, 2, 3])),
        ],
    );
    assert_same(&split);

    let int32 = tmp("int32.gguf");
    write_gguf(
        &int32,
        &[
            ("general.architecture", Val::Str("deepseek4")),
            ("deepseek4.block_count", Val::I32(43)),
        ],
    );
    assert_same(&int32);

    let ver2 = tmp("v2.gguf");
    let mut v2 = Vec::new();
    put_u32(&mut v2, 0x4655_4747);
    put_u32(&mut v2, 2);
    put_u64(&mut v2, 0);
    put_u64(&mut v2, 0);
    while v2.len() < 32 {
        v2.push(0);
    }
    fs::write(&ver2, v2).unwrap();
    assert_same(&ver2);

    let bad_magic = tmp("bad-magic.gguf");
    fs::write(&bad_magic, [0u8; 32]).unwrap();
    assert_same(&bad_magic);

    let tiny = tmp("tiny.gguf");
    fs::write(&tiny, [0u8; 16]).unwrap();
    assert_same(&tiny);
}

#[test]
fn identify_gguf_motif_no_slurp() {
    let path = tmp("identify-motif.gguf");
    write_gguf(&path, &[("general.architecture", Val::Str("motif3"))]);
    let id = ds4_core::identify_gguf(&path).unwrap();
    assert_eq!(id.shape.name, "Motif-3");
    assert_eq!(id.shape.model_id(), 3);
    assert_eq!(id.split_count, 0);
    let file_len = fs::metadata(&path).unwrap().len();
    assert!(file_len >= 32);
}

#[test]
fn identify_glm53_flash() {
    let path = tmp("glm53.gguf");
    write_gguf(&path, &[("general.architecture", Val::Str("glm5-next"))]);
    assert_same(&path);
    let id = ds4_core::identify_gguf(&path).unwrap();
    assert_eq!(id.shape.name, "GLM 5.3 Flash");
    assert_eq!(id.shape.family as u32, 6);
    assert_eq!(id.shape.variant as u32, 7);
    assert_eq!(id.shape.n_layer, 46);
    assert_eq!(id.shape.n_embd, 4096);
    assert_eq!(id.shape.n_vocab, 154880);
}

#[test]
fn identify_k2_horizon_375b() {
    let path = tmp("k2-horizon-375b.gguf");
    write_gguf(&path, &[("general.architecture", Val::Str("k2-horizon"))]);
    assert_same(&path);
    let id = ds4_core::identify_gguf(&path).expect("identify K2 Horizon 375B");
    assert_eq!(id.shape.name, "K2-Horizon 375B A23B");
    assert_eq!(id.shape.family, ds4_core::ModelFamily::ExaoneMoe);
    assert_eq!(id.shape.variant, ds4_core::Variant::K2Horizon375B);
    assert_eq!(id.shape.n_layer, 61);
    assert_eq!(id.shape.n_embd, 6144);
    assert_eq!(id.shape.n_vocab, 250624);
    assert_eq!(id.shape.n_head, 48);
    assert_eq!(id.shape.n_head_kv, 8);
    assert_eq!(id.shape.n_rot, 64);
    assert_eq!(id.shape.n_expert, 192);
    assert_eq!(id.shape.n_expert_used, 8);
    assert_eq!(id.shape.n_leading_dense, 3);
    assert_eq!(id.shape.n_nextn_predict, 0);
    assert_eq!(id.shape.n_swa, 0);
    assert!(id.shape.use_rope);
    assert!(!id.shape.use_qk_norm);
}

#[test]
fn get_u32_rejects_int32() {
    let path = tmp("u32-only.gguf");
    write_gguf(&path, &[("k", Val::I32(7))]);
    let g = ds4_core::GgufFile::open(&path).unwrap();
    assert_eq!(g.get_u32("k"), None);
}
