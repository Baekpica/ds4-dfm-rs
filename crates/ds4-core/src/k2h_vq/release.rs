//! `release.json` lists independent GGUF parts (not numbered splits) by role,
//! both required: `trunk` (58 expert parts and the body, the device copy) and
//! `embedding` (the Q8_0 token table, `resident-or-host-rows`: native uploads
//! it or reads one row per token). The embedding part is placed last so
//! native finds the token table as the inventory's last tensor. Native maps
//! the returned page-aligned shard plan; offsets stay file-backed.
//!
//! The tokenizer, template and configuration are source files kept beside
//! the parts and checked by SHA-256 (`ASSETS`).

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::ops::RangeInclusive;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{
    fail, q8_scale_finite, Result, ALIGN, ARCH, INDEX_BITS, Q8_0, Q8_BYTES, RECIPE, VARIANT,
};
use crate::gguf::{GgufError, GgufFile};
use crate::mapped::page_size;
use crate::tensors::{ShardPlan, TensorError, TensorInventory};

const MAX_METADATA: u64 = 64 * 1024 * 1024;
const RELEASE: &str = "release.json";
const FORMAT_VERSION: u64 = 1;
/// Pinned IFM/K2-Horizon-375B-A23B revision of every stored weight.
const SOURCE_REPO: &str = "IFM/K2-Horizon-375B-A23B";
const SOURCE_REVISION: &str = "d33e3ae45281865ebf9f044b12d3635b1d1e17fe";
/// max_position_embeddings (kv_contract.native_context).
const CONTEXT: u64 = 524_288;
/// MoE layers, one expert part each; layers 0-2 are dense.
const EXPERT_LAYERS: RangeInclusive<u64> = 3..=60;
/// Source files kept beside the parts. config.json and the tokenizer files
/// are SOURCE_REVISION's; chat_template.jinja is IFM's later official
/// template (main 7a17b47a74b9371e71ed64b4080192e01caa6174), which adds the
/// `render_text_content` macro (typed text parts) to the d33e3ae4 one.
const ASSETS: [(&str, &str); 5] = [
    (
        "config.json",
        "0daf184082ec39c3b24c5da6fcd5739f3176f0ad2c9e21fddaaea24b7400a583",
    ),
    (
        "tokenizer.json",
        "2fa69519ff1ee3ee55ac50927b00058e88b9eb90802fabaf3c6b05804601c53b",
    ),
    (
        "tokenizer_config.json",
        "581375b2ce8d56d59f609f08a7165365ac53b6543f3442ef0117b4951b0d747c",
    ),
    (
        "special_tokens_map.json",
        "e9be461abfcb63da71fb65db470e075682e7573e469cc658aeeb6632a6144ae1",
    ),
    (
        "chat_template.jinja",
        "5008a8535f850190e119cbb64dcf74c9483c8b483e8e3558795d71e1c3e31e10",
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Trunk,
    Embedding,
}

impl Role {
    fn parse(entry: &Value) -> Result<Self> {
        let (role, residency) = match text(entry, "role")? {
            "trunk" => (Self::Trunk, "resident"),
            "embedding" => (Self::Embedding, "resident-or-host-rows"),
            _ => return Err(fail("unknown part role")),
        };
        if text(entry, "residency")? != residency
            || entry.get("required").and_then(Value::as_bool) != Some(true)
        {
            return Err(fail("part residency/required"));
        }
        Ok(role)
    }

    fn metadata(self) -> &'static [u8] {
        match self {
            Self::Trunk => b"trunk",
            Self::Embedding => b"embedding",
        }
    }
}

fn json_bytes(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(GgufError::Io)?
        .take(MAX_METADATA + 1)
        .read_to_end(&mut bytes)
        .map_err(GgufError::Io)?;
    if bytes.len() as u64 > MAX_METADATA {
        return Err(fail("oversized release metadata"));
    }
    Ok(bytes)
}

fn json(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice(bytes).map_err(|_| fail("invalid release JSON"))
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or(fail("missing release string"))
}

fn number(value: &Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or(fail("missing release integer"))
}

fn array<'a>(value: &'a Value, key: &str) -> Result<&'a [Value]> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or(fail("missing release array"))
}

fn filename(name: &str) -> Result<()> {
    let mut parts = Path::new(name).components();
    if name.contains(['/', '\\'])
        || !matches!(parts.next(), Some(Component::Normal(_)))
        || parts.next().is_some()
    {
        return Err(fail("release path must be one filename"));
    }
    Ok(())
}

fn check_hash(bytes: &[u8], expected: &str) -> Result<()> {
    check_digest(&Sha256::digest(bytes), expected)
}

fn check_digest(digest: &[u8], expected: &str) -> Result<()> {
    if expected.len() != 64 || !expected.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(fail("invalid release SHA256"));
    }
    for (i, byte) in digest.iter().enumerate() {
        if u8::from_str_radix(&expected[2 * i..2 * i + 2], 16).ok() != Some(*byte) {
            return Err(fail("release SHA256 mismatch"));
        }
    }
    Ok(())
}

pub(super) fn is_release(root: &Path) -> bool {
    let Ok(bytes) = json_bytes(&root.join(RELEASE)) else {
        return false;
    };
    json(&bytes).is_ok_and(|v| v.get("format").and_then(Value::as_str) == Some("k2-horizon-vq8"))
}

/// A pinned source file of the release directory, checked by its SHA-256.
pub(super) fn asset(root: &Path, name: &str) -> Result<Vec<u8>> {
    let (_, sha) = ASSETS
        .iter()
        .find(|(asset, _)| *asset == name)
        .ok_or(fail("unknown release asset"))?;
    let bytes = json_bytes(&root.join(name)).map_err(|_| fail("missing release asset"))?;
    check_hash(&bytes, sha)?;
    Ok(bytes)
}

struct Part {
    name: String,
    role: Role,
    entry: Value,
}

/// An expert part's declared layer: `layer-LL-experts.gguf`, 13-bit hVQ in
/// container 1, no residual stage.
fn expert_part(name: &str, entry: &Value) -> Result<Option<u64>> {
    let Some(layer) = entry.get("layer") else {
        return Ok(None);
    };
    let layer = layer.as_u64().ok_or(fail("missing release integer"))?;
    if !EXPERT_LAYERS.contains(&layer)
        || name != format!("layer-{layer:02}-experts.gguf")
        || text(entry, "variant")? != "R13"
        || number(entry, "container_version")? != 1
        || number(entry, "index_bits")? != u64::from(INDEX_BITS)
        || entry.get("residual").and_then(Value::as_bool) != Some(false)
        || text(entry, "method")? != "hvq"
    {
        return Err(fail("expert part layer/stage"));
    }
    Ok(Some(layer))
}

/// The release entrypoint's parts: the trunk (expert layers, then the body,
/// as published), then the embedding part.
fn parts(root: &Path) -> Result<Vec<Part>> {
    let release = json(&json_bytes(&root.join(RELEASE))?)?;
    let kv = release
        .get("kv_contract")
        .ok_or(fail("missing kv_contract"))?;
    if text(&release, "format")?.as_bytes() != VARIANT
        || number(&release, "format_version")? != FORMAT_VERSION
        || text(&release, "recipe_id")?.as_bytes() != RECIPE
        || text(&release, "source_repo")? != SOURCE_REPO
        || text(&release, "source_revision")? != SOURCE_REVISION
        || text(&release, "entrypoint")? != RELEASE
        || text(&release, "architecture")?.as_bytes() != ARCH
        || number(kv, "native_context")? != CONTEXT
    {
        return Err(fail("release format/recipe/source revision"));
    }
    let mut names = HashSet::new();
    let mut layers = Vec::new();
    let mut trunk = Vec::new();
    let mut embedding = Vec::new();
    for entry in array(&release, "parts")? {
        let name = text(entry, "path")?;
        filename(name)?;
        let role = Role::parse(entry)?;
        if !name.ends_with(".gguf") || !names.insert(name.to_owned()) {
            return Err(fail("part identity/role"));
        }
        if let Some(layer) = expert_part(name, entry)? {
            if role != Role::Trunk {
                return Err(fail("expert part role"));
            }
            layers.push(layer);
        }
        let part = Part {
            name: name.to_owned(),
            role,
            entry: entry.clone(),
        };
        match role {
            Role::Trunk => trunk.push(part),
            Role::Embedding => embedding.push(part),
        }
    }
    // 58 expert parts in layer order and the body; one token table.
    if layers != EXPERT_LAYERS.collect::<Vec<_>>()
        || trunk.len() != layers.len() + 1
        || embedding.len() != 1
    {
        return Err(fail("missing/duplicate release parts"));
    }
    trunk.append(&mut embedding);
    // Every GGUF on disk is a declared part; an unlisted one is not this model.
    for item in std::fs::read_dir(root).map_err(GgufError::Io)? {
        let name = item.map_err(GgufError::Io)?.file_name();
        let name = name
            .into_string()
            .map_err(|_| fail("non-UTF8 release path"))?;
        if name.ends_with(".gguf") && !names.contains(&name) {
            return Err(fail("unexpected GGUF part"));
        }
    }
    Ok(trunk)
}

pub(super) fn first_part(root: &Path) -> Result<PathBuf> {
    Ok(root.join(&parts(root)?[0].name))
}

pub(super) fn part_count(root: &Path) -> Result<u32> {
    Ok(parts(root)?.len() as u32)
}

/// On-disk bytes of the trunk parts. Hashes are checked when the release
/// opens, not here.
pub(super) fn trunk_bytes(root: &Path) -> Result<u64> {
    parts(root)?
        .iter()
        .filter(|p| p.role == Role::Trunk)
        .try_fold(0u64, |total, part| {
            let len = std::fs::metadata(root.join(&part.name))
                .map_err(GgufError::Io)?
                .len();
            Ok(total.saturating_add(len))
        })
}

const MAX_VERIFIERS: usize = 16;
/// File mtimes are coarse (one jiffy); only entries this old are cached, so
/// any later rewrite is guaranteed a different mtime.
const SETTLED: Duration = Duration::from_secs(2);

type Stamp = Vec<(OsString, u64, SystemTime)>;

/// Inventories already verified in this process, keyed by directory and the
/// size/mtime of every entry. Each load path (validation, binding, serving
/// identity) reopens the directory; one check is enough.
static VERIFIED: Mutex<Vec<(PathBuf, Stamp, TensorInventory)>> = Mutex::new(Vec::new());

fn settled(stamp: &Stamp) -> bool {
    let Some(limit) = SystemTime::now().checked_sub(SETTLED) else {
        return false;
    };
    stamp.iter().all(|(_, _, modified)| *modified <= limit)
}

fn stamp(root: &Path) -> Result<Stamp> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(root).map_err(GgufError::Io)? {
        let entry = entry.map_err(GgufError::Io)?;
        let meta = entry.metadata().map_err(GgufError::Io)?;
        entries.push((
            entry.file_name(),
            meta.len(),
            meta.modified().map_err(GgufError::Io)?,
        ));
    }
    entries.sort();
    Ok(entries)
}

pub(super) fn open_dir(root: &Path) -> Result<TensorInventory> {
    let key = root.canonicalize().map_err(GgufError::Io)?;
    let before = stamp(root)?;
    let known = VERIFIED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|(k, s, _)| *k == key && *s == before)
        .map(|(_, _, inventory)| inventory.clone());
    if let Some(inventory) = known {
        return Ok(inventory);
    }
    let inventory = verify_dir(root)?;
    // A part rewritten during verification must not be cached as verified.
    if settled(&before) && stamp(root)? == before {
        VERIFIED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((key, before, inventory.clone()));
    }
    Ok(inventory)
}

/// Hash and parse one part with part-relative offsets.
fn verify_part(root: &Path, part: &Part) -> Result<(PathBuf, u64, TensorInventory)> {
    let path = root.join(&part.name);
    let g = GgufFile::open(&path)?;
    random_access(g.as_bytes());
    let layer = part.entry.get("layer").and_then(Value::as_u64);
    if g.get_string("k2_horizon_vq.part_role") != Some(part.role.metadata())
        || g.get_string("k2_horizon_vq.source_revision") != Some(SOURCE_REVISION.as_bytes())
        || g.get_u32("k2_horizon_vq.layer").map(u64::from) != layer
    {
        return Err(fail("part role/source revision/layer mismatch"));
    }
    let inv = super::streamed(|| TensorInventory::from_file(&path, &g))?;
    if inv.shards.len() != 1 || inv.alignment != ALIGN {
        return Err(fail("part is not one aligned GGUF"));
    }
    // The embedding part holds the token table alone.
    let embedding = inv.tensors.len() == 1 && inv.tensors[0].name == super::layout::EMBEDDING;
    if (part.role == Role::Embedding) != embedding
        || inv
            .tensors
            .iter()
            .any(|t| t.name == super::layout::EMBEDDING)
            != embedding
    {
        return Err(fail("part tensors/embedding table"));
    }
    if number(&part.entry, "bytes")? != g.as_bytes().len() as u64 {
        return Err(fail("release file size mismatch"));
    }
    let q8: Vec<(u64, u64)> = inv
        .tensors
        .iter()
        .filter(|t| t.typ == Q8_0)
        .map(|t| (t.abs_offset, t.bytes))
        .collect();
    let digest = part_digest(&path, g.as_bytes(), Scan::new(&q8))?;
    check_digest(&digest, text(&part.entry, "sha256")?)?;
    let size = g.as_bytes().len() as u64;
    drop(g);
    drop_cache(&path);
    Ok((path, size, inv))
}

/// Parts hash through O_DIRECT reads: streaming ~87 GB through the page
/// cache fragments the free memory the native 2 MiB device pages come from
/// (GB10 UMA). The mapped bytes are the fallback. The same pass checks the
/// Q8_0 scales `Scan` covers.
fn part_digest(path: &Path, mapped: &[u8], mut scan: Scan) -> Result<Vec<u8>> {
    #[cfg(target_os = "linux")]
    if let Some(digest) = direct_digest(path, &mut scan)? {
        return Ok(digest);
    }
    scan.feed(0, mapped)?;
    Ok(Sha256::digest(mapped).to_vec())
}

/// Q8_0 scale checks as a part streams by (spans in part offsets). A
/// block's two scale bytes may straddle two chunks; the previous chunk's
/// last byte carries.
struct Scan<'a> {
    q8: &'a [(u64, u64)],
    tail: u8,
}

impl<'a> Scan<'a> {
    fn new(q8: &'a [(u64, u64)]) -> Self {
        Self { q8, tail: 0 }
    }

    fn feed(&mut self, pos: u64, chunk: &[u8]) -> Result<()> {
        let Some(&last) = chunk.last() else {
            return Ok(());
        };
        let end = pos + chunk.len() as u64;
        // A block starting on the previous chunk's last byte waited.
        let from = pos.saturating_sub(1);
        let byte = |at: u64| {
            if at >= pos {
                chunk[(at - pos) as usize]
            } else {
                self.tail
            }
        };
        for &(start, bytes) in self.q8 {
            let stop = start + bytes;
            if stop <= from || start >= end {
                continue;
            }
            let mut at = start + from.saturating_sub(start).div_ceil(Q8_BYTES) * Q8_BYTES;
            while at < stop && at + 2 <= end {
                if !q8_scale_finite(&[byte(at), byte(at + 1)]) {
                    return Err(fail("nonfinite Q8_0 scale"));
                }
                at += Q8_BYTES;
            }
        }
        self.tail = last;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn direct_digest(path: &Path, scan: &mut Scan) -> Result<Option<Vec<u8>>> {
    use std::os::unix::fs::OpenOptionsExt;
    const DIRECT_ALIGN: usize = 4096;
    const DIRECT_CHUNK: usize = 16 << 20;
    let Ok(mut file) = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    else {
        return Ok(None);
    };
    let mut raw = vec![0u8; DIRECT_CHUNK + DIRECT_ALIGN];
    let start = raw.as_ptr().align_offset(DIRECT_ALIGN);
    let buf = &mut raw[start..start + DIRECT_CHUNK];
    let mut hasher = Sha256::new();
    let mut pos = 0u64;
    loop {
        match file.read(buf) {
            Ok(0) => return Ok(Some(hasher.finalize().to_vec())),
            Ok(n) => {
                hasher.update(&buf[..n]);
                scan.feed(pos, &buf[..n])?;
                pos += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return Ok(None),
        }
    }
}

/// Validation reads headers, gains, books and padding through the map;
/// readahead around each would cache ~200 MiB per part.
#[cfg(target_os = "linux")]
fn random_access(bytes: &[u8]) {
    unsafe {
        libc::madvise(
            bytes.as_ptr() as *mut libc::c_void,
            bytes.len(),
            libc::MADV_RANDOM,
        )
    };
}

#[cfg(not(target_os = "linux"))]
fn random_access(_: &[u8]) {}

/// Validation still faults headers, books and scales through the map;
/// return those pages once a part is verified.
#[cfg(target_os = "linux")]
fn drop_cache(path: &Path) {
    use std::os::fd::AsRawFd;
    if let Ok(file) = File::open(path) {
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_cache(_: &Path) {}

fn verify_dir(root: &Path) -> Result<TensorInventory> {
    let parts = parts(root)?;
    let page = page_size();
    let mut result = TensorInventory {
        shards: Vec::new(),
        tensors: Vec::new(),
        data_pos: 0,
        alignment: ALIGN,
        page,
    };
    // Parts hash independently (~87 GB); workers pull parts in release
    // order and assembly below stays sequential, so offsets and the first
    // reported error match a serial pass.
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(MAX_VERIFIERS)
        .min(parts.len());
    let mut verified: Vec<Option<Result<(PathBuf, u64, TensorInventory)>>> =
        parts.iter().map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(part) = parts.get(i) else {
                            return done;
                        };
                        done.push((i, verify_part(root, part)));
                    }
                })
            })
            .collect();
        for handle in handles {
            for (i, part) in handle.join().expect("K2-Horizon VQ part verifier panicked") {
                verified[i] = Some(part);
            }
        }
    });
    let mut names = HashSet::new();
    let mut base = 0u64;
    for part in verified {
        let (path, size, mut inv) = part.expect("every part is verified")?;
        let shard = result.shards.len() as u32;
        if shard == 0 {
            result.data_pos = inv.data_pos;
        }
        for tensor in &mut inv.tensors {
            if !names.insert(tensor.name.clone()) {
                return Err(fail("duplicate tensor across parts"));
            }
            tensor.shard = shard;
            tensor.abs_offset = tensor
                .abs_offset
                .checked_add(base)
                .ok_or(TensorError::Overflow)?;
        }
        result.shards.push(ShardPlan { path, size, base });
        result.tensors.extend(inv.tensors);
        let span = size.checked_add(page - 1).ok_or(TensorError::Overflow)? / page * page;
        base = base.checked_add(span).ok_or(TensorError::Overflow)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expert_parts_are_r13_hvq() {
        let r13 = json!({"layer": 3, "variant": "R13", "container_version": 1, "index_bits": 13,
            "residual": false, "method": "hvq"});
        assert_eq!(expert_part("layer-03-experts.gguf", &r13).unwrap(), Some(3));
        assert_eq!(
            expert_part("body-nonexpert.gguf", &json!({})).unwrap(),
            None
        );
        assert!(expert_part("layer-04-experts.gguf", &r13).is_err());
        for (key, value) in [
            ("variant", json!("R12")),
            ("container_version", json!(2)),
            ("index_bits", json!(12)),
            ("residual", json!(true)),
            ("method", json!("ivq")),
            ("layer", json!(2)),
            ("layer", json!(61)),
        ] {
            let mut bad = r13.clone();
            bad[key] = value;
            let layer = bad["layer"].as_u64().unwrap();
            assert!(
                expert_part(&format!("layer-{layer:02}-experts.gguf"), &bad).is_err(),
                "accepted {key}"
            );
        }
    }

    #[test]
    fn roles_pin_residency() {
        let embedding = json!({"role": "embedding", "residency": "resident-or-host-rows",
            "required": true});
        assert_eq!(Role::parse(&embedding).unwrap(), Role::Embedding);
        for (key, value) in [
            ("residency", json!("ssd-rows")),
            ("required", json!(false)),
            ("role", json!("vision")),
        ] {
            let mut bad = embedding.clone();
            bad[key] = value;
            assert!(Role::parse(&bad).is_err(), "accepted {key}");
        }
    }

    #[test]
    fn assets_are_pinned() {
        let dir = std::env::temp_dir().join(format!("k2h-assets-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("release.json"), b"{}").unwrap();
        assert!(asset(&dir, "config.json").is_err());
        assert!(asset(&dir, "tokenizer.json").is_err());
        assert!(asset(&dir, "release.json").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // A NaN/inf scale anywhere in a Q8_0 span must be found under any
    // chunking, including a scale that straddles a chunk boundary.
    #[test]
    fn scan_finds_straddling_scales() {
        let q8 = [(128, 3 * Q8_BYTES)];
        let clean = vec![0u8; 512];
        let last = 128 + 2 * Q8_BYTES as usize;
        let mut bad = clean.clone();
        bad[last + 1] = 0x7c;
        for chunk in [1, 2, 3, 7, 34, last + 1, 512] {
            let scan = |data: &[u8]| -> Result<()> {
                let mut scan = Scan::new(&q8);
                for (i, part) in data.chunks(chunk).enumerate() {
                    scan.feed((i * chunk) as u64, part)?;
                }
                Ok(())
            };
            assert!(scan(&clean).is_ok(), "chunk {chunk}");
            assert!(scan(&bad).is_err(), "chunk {chunk}");
        }
    }
}
