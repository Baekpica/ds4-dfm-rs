//! `release.json` lists independent GGUF parts (not numbered splits) by role:
//! `trunk` (device copy) and `embedding` (BF16 token rows read from SSD) are
//! required; the `drafter` (MTP, device copy) joins only when speculative
//! decoding is requested and the `vision` sidecar is never part of the device
//! inventory. The embedding part is placed last so native finds the token
//! table as the inventory's last tensor. Native maps the returned
//! page-aligned shard plan; offsets stay file-backed.
//!
//! The release ships no tokenizer, template or configuration: those are the
//! source checkpoint's files at the pinned revision, kept beside the parts and
//! checked by SHA-256 (`ASSETS`).

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{
    fail, q4_scales_finite, q8_scale_finite, Result, ALIGN, INDEX_BITS, Q4_BYTES, Q4_K, Q8_0,
    Q8_BYTES, RECIPE, VARIANT,
};
use crate::gguf::{GgufError, GgufFile};
use crate::mapped::page_size;
use crate::tensors::{ShardPlan, TensorError, TensorInventory};

const MAX_METADATA: u64 = 64 * 1024 * 1024;
const RELEASE: &str = "release.json";
const FORMAT_VERSION: u64 = 1;
/// Pinned thinkingmachines/Inkling-2-Small-MXFP8 revision of every stored weight.
const SOURCE_REPO: &str = "thinkingmachines/Inkling-2-Small-MXFP8";
const SOURCE_REVISION: &str = "086a047e8b7ff2ac902dd52355328ddb66b16914";
const CONTEXT: u64 = 1_048_576;
const EXPERT_LAYERS: u64 = 48;
const GLOBAL_LAYERS: [u64; 8] = [5, 11, 17, 23, 29, 35, 41, 47];
/// The source files the runtime reads, as published at SOURCE_REVISION.
const ASSETS: [(&str, &str); 5] = [
    (
        "config.json",
        "7da68011c7c85af5e040114d3ebd5c8fdc8ce9030851a5706ebe88c379632405",
    ),
    (
        "tokenizer.json",
        "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6",
    ),
    (
        "chat_template.jinja",
        "e9cfd8e75d1272633c2e72b3cc76e99567e2bc8912b960483aed483ad35d54c7",
    ),
    (
        "tokenizer_config.json",
        "6addc2bf8dc756dea8edd083a4ea3b67a66b8f513e9745a36e4a7cc0cc9f55ed",
    ),
    (
        "processor_config.json",
        "b326f9a5d49c037c463045bec4d80229f5463048df7d6de95b597667888228ae",
    ),
];

/// Device-inventory features: the drafter part joins the trunk on request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Feature {
    Trunk,
    Draft,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Role {
    Trunk,
    Embedding,
    Vision,
    Drafter,
}

impl Role {
    fn parse(entry: &Value) -> Result<Self> {
        let (role, residency, required) = match text(entry, "role")? {
            "trunk" => (Self::Trunk, "resident", true),
            "embedding" => (Self::Embedding, "ssd-rows", true),
            "vision" => (Self::Vision, "mmap-on-demand", false),
            "drafter" => (Self::Drafter, "resident", false),
            _ => return Err(fail("unknown part role")),
        };
        if text(entry, "residency")? != residency
            || entry.get("required").and_then(Value::as_bool) != Some(required)
        {
            return Err(fail("part residency/required"));
        }
        Ok(role)
    }

    fn metadata(self) -> &'static [u8] {
        match self {
            Self::Trunk => b"trunk",
            Self::Embedding => b"embedding",
            Self::Vision => b"vision",
            Self::Drafter => b"drafter",
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
    json(&bytes).is_ok_and(|v| v.get("format").and_then(Value::as_str) == Some("inkling2-vq8"))
}

/// A pinned source file of the release directory, checked by its SHA-256.
pub(super) fn asset(root: &Path, name: &str) -> Result<Vec<u8>> {
    let (_, sha) = ASSETS
        .iter()
        .find(|(asset, _)| *asset == name)
        .ok_or(fail("unknown release asset"))?;
    let bytes = json_bytes(&root.join(name)).map_err(|_| {
        fail("missing source asset: copy it from thinkingmachines/Inkling-2-Small-MXFP8@086a047e")
    })?;
    check_hash(&bytes, sha)?;
    Ok(bytes)
}

struct Part {
    name: String,
    role: Role,
    entry: Value,
}

/// An expert part's declared layer: `layer-LL-experts.gguf`, 12-bit hVQ in
/// container 1, no residual stage.
fn expert_part(name: &str, entry: &Value) -> Result<Option<u64>> {
    let Some(layer) = entry.get("layer") else {
        return Ok(None);
    };
    let layer = layer.as_u64().ok_or(fail("missing release integer"))?;
    if layer >= EXPERT_LAYERS
        || name != format!("layer-{layer:02}-experts.gguf")
        || text(entry, "variant")? != "R12"
        || number(entry, "container_version")? != 1
        || number(entry, "index_bits")? != u64::from(INDEX_BITS)
        || entry.get("residual").and_then(Value::as_bool) != Some(false)
        || text(entry, "method")? != "hvq"
    {
        return Err(fail("expert part layer/stage"));
    }
    Ok(Some(layer))
}

/// The verified release entrypoint and the parts the feature selects: the
/// trunk (expert layers, then the body, as published), the drafter on
/// request, then the embedding part.
fn parts(root: &Path, feature: Feature) -> Result<Vec<Part>> {
    Ok(release_parts(root, feature)?.0)
}

/// The release lists a vision part (unverified: the quote's question).
pub(super) fn has_vision(root: &Path) -> bool {
    release_parts(root, Feature::Trunk).is_ok_and(|(_, vision)| vision.is_some())
}

/// The vision part, verified, when the release ships one (mapped at open,
/// read only when media arrives).
pub(super) fn vision_path(root: &Path) -> Result<Option<PathBuf>> {
    match release_parts(root, Feature::Trunk)?.1 {
        Some(part) => Ok(Some(verify_part(root, &part)?.0)),
        None => Ok(None),
    }
}

fn release_parts(root: &Path, feature: Feature) -> Result<(Vec<Part>, Option<Part>)> {
    let release = json(&json_bytes(&root.join(RELEASE))?)?;
    let kv = release
        .get("kv_contract")
        .ok_or(fail("missing kv_contract"))?;
    let globals: Vec<u64> = array(kv, "global_layers")?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    if text(&release, "format")?.as_bytes() != VARIANT
        || number(&release, "format_version")? != FORMAT_VERSION
        || text(&release, "recipe_id")?.as_bytes() != RECIPE
        || text(&release, "source_repo")? != SOURCE_REPO
        || text(&release, "source_revision")? != SOURCE_REVISION
        || text(&release, "entrypoint")? != RELEASE
        || text(&release, "architecture")? != "inkling2_vq"
        || number(kv, "native_context")? != CONTEXT
        || globals != GLOBAL_LAYERS
    {
        return Err(fail("release format/recipe/source revision"));
    }
    let mut names = HashSet::new();
    let mut layers = Vec::new();
    let mut counts = HashMap::new();
    let mut selected = Vec::new();
    let mut embedding = None;
    let mut vision = None;
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
        *counts.entry(role).or_insert(0u32) += 1;
        let part = Part {
            name: name.to_owned(),
            role,
            entry: entry.clone(),
        };
        match role {
            Role::Trunk => selected.push(part),
            Role::Drafter if feature == Feature::Draft => selected.push(part),
            Role::Embedding => embedding = Some(part),
            Role::Vision => vision = Some(part),
            Role::Drafter => {}
        }
    }
    // 48 expert parts and the body; one table; at most one of each sidecar.
    if layers != (0..EXPERT_LAYERS).collect::<Vec<_>>()
        || counts.get(&Role::Trunk) != Some(&(layers.len() as u32 + 1))
        || counts.get(&Role::Embedding) != Some(&1)
        || counts.get(&Role::Vision).copied().unwrap_or(0) > 1
        || counts.get(&Role::Drafter).copied().unwrap_or(0) > 1
        || (feature == Feature::Draft && !counts.contains_key(&Role::Drafter))
    {
        return Err(fail("missing/duplicate release parts"));
    }
    selected.extend(embedding);
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
    Ok((selected, vision))
}

pub(super) fn first_part(root: &Path) -> Result<PathBuf> {
    Ok(root.join(&parts(root, Feature::Trunk)?[0].name))
}

pub(super) fn part_count(root: &Path) -> Result<u32> {
    Ok(parts(root, Feature::Trunk)?.len() as u32)
}

/// On-disk bytes of the device-copied parts (trunk, and the drafter with
/// `Feature::Draft`). Hashes are checked when the release opens, not here.
pub(super) fn resident_bytes(root: &Path, feature: Feature) -> Result<u64> {
    parts(root, feature)?
        .iter()
        .filter(|p| p.role != Role::Embedding)
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

/// Inventories already verified in this process, keyed by directory,
/// feature and the size/mtime of every entry. Each load path (validation,
/// binding, serving identity) reopens the directory; one check is enough.
static VERIFIED: Mutex<Vec<(PathBuf, Feature, Stamp, TensorInventory)>> = Mutex::new(Vec::new());

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

pub(super) fn open_dir(root: &Path, feature: Feature) -> Result<TensorInventory> {
    let key = root.canonicalize().map_err(GgufError::Io)?;
    let before = stamp(root)?;
    let known = VERIFIED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|(k, f, s, _)| *k == key && *f == feature && *s == before)
        .map(|(_, _, _, inventory)| inventory.clone());
    if let Some(inventory) = known {
        return Ok(inventory);
    }
    let inventory = verify_dir(root, feature)?;
    // A part rewritten during verification must not be cached as verified.
    if settled(&before) && stamp(root)? == before {
        VERIFIED.lock().unwrap_or_else(|e| e.into_inner()).push((
            key,
            feature,
            before,
            inventory.clone(),
        ));
    }
    Ok(inventory)
}

/// Hash and parse one part with part-relative offsets.
fn verify_part(root: &Path, part: &Part) -> Result<(PathBuf, u64, TensorInventory)> {
    let path = root.join(&part.name);
    let g = GgufFile::open(&path)?;
    random_access(g.as_bytes());
    let layer = part.entry.get("layer").and_then(Value::as_u64);
    if g.get_string("inkling2_vq.part_role") != Some(part.role.metadata())
        || g.get_string("inkling2_vq.source_revision") != Some(SOURCE_REVISION.as_bytes())
        || g.get_u32("inkling2_vq.layer").map(u64::from) != layer
        || (layer.is_some() && g.get_u32("inkling2_vq.index_bits") != Some(u32::from(INDEX_BITS)))
    {
        return Err(fail("part role/source revision/layer mismatch"));
    }
    let inv = super::streamed(|| TensorInventory::from_file(&path, &g))?;
    if inv.shards.len() != 1 || inv.alignment != ALIGN {
        return Err(fail("part is not one aligned GGUF"));
    }
    // The embedding part holds the token table alone.
    let embedding = inv.tensors.len() == 1 && inv.tensors[0].name == super::EMBEDDING;
    if (part.role == Role::Embedding) != embedding
        || inv.tensors.iter().any(|t| t.name == super::EMBEDDING) != embedding
    {
        return Err(fail("part tensors/embedding table"));
    }
    if number(&part.entry, "bytes")? != g.as_bytes().len() as u64 {
        return Err(fail("release file size mismatch"));
    }
    let spans = |typ| -> Vec<(u64, u64)> {
        inv.tensors
            .iter()
            .filter(|t| t.typ == typ)
            .map(|t| (t.abs_offset, t.bytes))
            .collect()
    };
    let (q4, q8) = (spans(Q4_K), spans(Q8_0));
    let digest = part_digest(&path, g.as_bytes(), Scan::new(&q4, &q8))?;
    check_digest(&digest, text(&part.entry, "sha256")?)?;
    let size = g.as_bytes().len() as u64;
    drop(g);
    drop_cache(&path);
    Ok((path, size, inv))
}

/// Parts hash through O_DIRECT reads: streaming ~110 GB through the page
/// cache fragments the free memory the native 2 MiB device pages come from
/// (GB10 UMA). The mapped bytes are the fallback. The same pass checks the
/// Q4_K and Q8_0 scales `Scan` covers.
fn part_digest(path: &Path, mapped: &[u8], mut scan: Scan) -> Result<Vec<u8>> {
    #[cfg(target_os = "linux")]
    if let Some(digest) = direct_digest(path, &mut scan)? {
        return Ok(digest);
    }
    scan.feed(0, mapped)?;
    Ok(Sha256::digest(mapped).to_vec())
}

/// Quant-scale checks as a part streams by (spans in part offsets). A
/// block's scale bytes may straddle two chunks; the previous chunk's last
/// three bytes carry.
struct Scan<'a> {
    q4: &'a [(u64, u64)],
    q8: &'a [(u64, u64)],
    tail: [u8; 3],
}

impl<'a> Scan<'a> {
    fn new(q4: &'a [(u64, u64)], q8: &'a [(u64, u64)]) -> Self {
        Self {
            q4,
            q8,
            tail: [0; 3],
        }
    }

    fn feed(&mut self, pos: u64, chunk: &[u8]) -> Result<()> {
        let end = pos + chunk.len() as u64;
        // Blocks starting in the previous chunk's last three bytes waited.
        let from = pos.saturating_sub(3);
        let tail = self.tail;
        let byte = |at: u64| {
            if at >= pos {
                chunk[(at - pos) as usize]
            } else {
                tail[(3 - (pos - at)) as usize]
            }
        };
        for &(start, bytes) in self.q4 {
            let stop = start + bytes;
            if stop <= from || start >= end {
                continue;
            }
            let mut at = start + from.saturating_sub(start).div_ceil(Q4_BYTES) * Q4_BYTES;
            while at < stop && at + 4 <= end {
                if !q4_scales_finite(&[byte(at), byte(at + 1), byte(at + 2), byte(at + 3)]) {
                    return Err(fail("nonfinite Q4_K scale"));
                }
                at += Q4_BYTES;
            }
        }
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
        // The last three bytes of tail ++ chunk.
        let take = chunk.len().min(3);
        self.tail.copy_within(take.., 0);
        self.tail[3 - take..].copy_from_slice(&chunk[chunk.len() - take..]);
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

fn verify_dir(root: &Path, feature: Feature) -> Result<TensorInventory> {
    let parts = parts(root, feature)?;
    let page = page_size();
    let mut result = TensorInventory {
        shards: Vec::new(),
        tensors: Vec::new(),
        data_pos: 0,
        alignment: ALIGN,
        page,
    };
    // Parts hash independently (~110 GB); workers pull parts in release
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
            for (i, part) in handle.join().expect("Inkling-2 VQ part verifier panicked") {
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
    fn expert_parts_are_r12_hvq() {
        let r12 = json!({"layer": 3, "variant": "R12", "container_version": 1, "index_bits": 12,
            "residual": false, "method": "hvq"});
        assert_eq!(expert_part("layer-03-experts.gguf", &r12).unwrap(), Some(3));
        assert_eq!(
            expert_part("body-nonexpert.gguf", &json!({})).unwrap(),
            None
        );
        assert!(expert_part("layer-04-experts.gguf", &r12).is_err());
        for (key, value) in [
            ("variant", json!("R13")),
            ("container_version", json!(2)),
            ("index_bits", json!(13)),
            ("residual", json!(true)),
            ("method", json!("ivq")),
            ("layer", json!(48)),
        ] {
            let mut bad = r12.clone();
            bad[key] = value;
            let layer = bad["layer"].as_u64().unwrap();
            assert!(
                expert_part(&format!("layer-{layer:02}-experts.gguf"), &bad).is_err(),
                "accepted {key}"
            );
        }
    }

    #[test]
    fn assets_are_pinned() {
        let dir = std::env::temp_dir().join(format!("ink2-assets-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        assert!(asset(&dir, "config.json").is_err());
        assert!(asset(&dir, "tokenizer.json").is_err());
        assert!(asset(&dir, "release.json").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
