//! C↔Rust weights_bind name catalog + bind-plan check/match.

use ds4_core::{
    bind_dspark_names, bind_mtp_names, bind_names, catalog_from_bind_name, dump_bind_check_oracle,
    dump_bind_lookup_tapes, dump_bind_match_oracle, dump_bind_names, dump_bind_names_variant,
    dump_bind_support, expected_compress_ratio, BindNeed, BindPlan, SupportCatalog,
    TensorInventory, Variant, SHAPE_FLASH, SHAPE_MOTIF3,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_BIND_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/bind_c_oracle")
}

fn lookup_oracle() -> PathBuf {
    if let Ok(p) = std::env::var("DS4_BIND_LOOKUP_C_ORACLE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/bind_lookup_c_oracle")
}

fn require_oracle() -> PathBuf {
    let p = oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/bind_c_oracle (missing {})",
        p.display()
    );
    p
}

fn c_out(args: &[&str]) -> String {
    let out = Command::new(require_oracle())
        .args(args)
        .output()
        .expect("run bind_c_oracle");
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

fn write_gguf(path: &Path, names: &[&str]) {
    let mut buf = Vec::new();
    put_u32(&mut buf, 0x4655_4747);
    put_u32(&mut buf, 3);
    put_u64(&mut buf, names.len() as u64);
    put_u64(&mut buf, 1);
    put_str(&mut buf, "general.alignment");
    put_u32(&mut buf, 4);
    put_u32(&mut buf, 32);
    for (i, name) in names.iter().enumerate() {
        put_str(&mut buf, name);
        put_u32(&mut buf, 1);
        put_u64(&mut buf, 8);
        put_u32(&mut buf, 0);
        put_u64(&mut buf, (i as u64) * 32);
    }
    let align = 32u64;
    let rem = (buf.len() as u64) % align;
    let data_pos = if rem == 0 {
        buf.len()
    } else {
        buf.len() + (align - rem) as usize
    };
    buf.resize(data_pos + names.len() * 32, 0);
    fs::write(path, buf).unwrap();
}

#[test]
fn bind_names_match_c() {
    let rust = dump_bind_names();
    let c = c_out(&["names"]);
    if rust != c {
        panic!(
            "bind names mismatch (rust {} bytes, c {} bytes)",
            rust.len(),
            c.len()
        );
    }
}

#[test]
fn bind_support_match_c() {
    assert_eq!(dump_bind_support(), c_out(&["support"]));
}

#[test]
fn support_catalog_names_are_deepseek_only() {
    assert_eq!(
        catalog_from_bind_name("mtp-flash"),
        Some((Some(SupportCatalog::Mtp), Variant::Flash))
    );
    assert_eq!(
        catalog_from_bind_name("dspark-pro"),
        Some((Some(SupportCatalog::Dspark), Variant::Pro))
    );
    assert!(catalog_from_bind_name("mtp-motif3").is_none());
    assert!(catalog_from_bind_name("dspark-solar-open2").is_none());
    assert_eq!(bind_mtp_names().len(), 32);
    assert_eq!(bind_dspark_names().len(), 9 + 3 * 24);
}

#[test]
fn step_mtp_catalog_is_complete() {
    assert_eq!(
        catalog_from_bind_name("mtp-step35"),
        Some((Some(SupportCatalog::Mtp), Variant::Step37Flash))
    );
    let dump = dump_bind_names_variant("mtp-step35").expect("Step MTP catalog");
    let actual: BTreeSet<_> = dump
        .lines()
        .filter_map(|line| line.strip_prefix("NAME "))
        .map(|line| line.split_whitespace().next().unwrap())
        .collect();
    let expected: BTreeSet<_> = include_str!("../../../tests/fixtures/step37/mtp.tsv")
        .lines()
        .map(|line| line.split('\t').next().unwrap())
        .collect();
    assert_eq!(actual, expected);
    assert!(dump.ends_with("COUNT n=55 req=55 opt=0\n"));
    assert!(catalog_from_bind_name("dspark-step35").is_none());
}

#[test]
fn bind_check_match_c() {
    assert_eq!(dump_bind_check_oracle(), c_out(&["check"]));
}

#[test]
fn bind_match_match_c() {
    assert_eq!(dump_bind_match_oracle(), c_out(&["match"]));
}

#[test]
fn bind_lookup_tapes_match_c() {
    let p = lookup_oracle();
    assert!(
        p.exists(),
        "build the C oracle first: make tests/parity/bind_lookup_c_oracle (missing {})",
        p.display()
    );
    let out = Command::new(&p).output().expect("run bind_lookup_c_oracle");
    assert!(
        out.status.success(),
        "lookup oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let c = String::from_utf8(out.stdout).expect("oracle utf8");
    assert_eq!(dump_bind_lookup_tapes(), c);
}

#[test]
fn flash_compress_ratio_matches_c() {
    assert_eq!(expected_compress_ratio(Variant::Flash, 43, 0), 0);
    assert_eq!(expected_compress_ratio(Variant::Flash, 43, 1), 0);
    assert_eq!(expected_compress_ratio(Variant::Flash, 43, 2), 4);
    assert_eq!(expected_compress_ratio(Variant::Flash, 43, 3), 128);
    assert_eq!(expected_compress_ratio(Variant::Pro, 61, 0), 128);
    assert_eq!(expected_compress_ratio(Variant::Pro, 61, 1), 128);
    assert_eq!(expected_compress_ratio(Variant::Pro, 61, 2), 4);
}

#[test]
fn resolve_reports_missing_required() {
    let dir = std::env::temp_dir().join(format!("ds4-bind-miss-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("empty.gguf");
    write_gguf(&path, &["token_embd.weight"]);
    let inv = TensorInventory::open(&path).unwrap();
    let plan = BindPlan::resolve(SHAPE_MOTIF3, &inv);
    let missing = plan.missing_required();
    assert!(missing.contains(&"output_norm.weight"));
    assert!(plan.check().is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn resolve_accepts_complete_flash_inventory() {
    let names = bind_names(&SHAPE_FLASH);
    let required: Vec<&str> = names
        .iter()
        .filter(|n| n.need == BindNeed::Required)
        .map(|n| n.name.as_str())
        .collect();
    assert!(required.contains(&"token_embd.weight"));
    assert!(required.contains(&"output_hc_base.weight"));
    let dir = std::env::temp_dir().join(format!("ds4-bind-ok-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("flash.gguf");
    write_gguf(&path, &required);
    let inv = TensorInventory::open(&path).unwrap();
    let plan = BindPlan::resolve(SHAPE_FLASH, &inv);
    plan.check().expect("complete flash inventory");
    assert!(plan.missing_required().is_empty());
    assert!(inv.find("token_embd.weight").is_some());
    let embd = plan
        .slots
        .iter()
        .find(|s| s.name == "token_embd.weight")
        .expect("embd slot");
    assert_eq!(embd.index, Some(0));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn resolve_mtp_reports_missing_required() {
    let dir = std::env::temp_dir().join(format!("ds4-mtp-miss-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("empty.gguf");
    write_gguf(&path, &["mtp.0.hc_head_base.weight"]);
    let inv = TensorInventory::open(&path).unwrap();
    let plan = BindPlan::resolve_mtp(SHAPE_FLASH, &inv);
    let missing = plan.missing_required();
    assert!(missing.contains(&"mtp.0.norm.weight"));
    assert!(missing.contains(&"mtp.0.ffn_down_shexp.weight"));
    assert!(!missing.contains(&"token_embd.weight"));
    assert!(plan.check().is_err());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn resolve_dspark_accepts_complete_inventory() {
    let names = bind_dspark_names();
    let required: Vec<&str> = names.iter().map(|n| n.name.as_str()).collect();
    let dir = std::env::temp_dir().join(format!("ds4-dspark-ok-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dspark.gguf");
    write_gguf(&path, &required);
    let inv = TensorInventory::open(&path).unwrap();
    let plan = BindPlan::resolve_dspark(SHAPE_FLASH, &inv);
    plan.check().expect("complete dspark inventory");
    assert!(plan.missing_required().is_empty());
    assert_eq!(plan.slots.len(), names.len());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn motif_leading_dense_uses_dense_ffn_names() {
    let names = bind_names(&SHAPE_MOTIF3);
    assert!(names
        .iter()
        .any(|n| n.name == "blk.0.ffn_gate.weight" && n.need == BindNeed::Required));
    assert!(names
        .iter()
        .any(|n| n.name == "blk.2.ffn_gate_exps.weight" && n.need == BindNeed::Required));
    assert!(names
        .iter()
        .any(|n| n.name == "mtp.0.embed_norm.weight" && n.need == BindNeed::Required));
}
