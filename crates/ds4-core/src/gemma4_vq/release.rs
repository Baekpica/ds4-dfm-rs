//! `release.json` lists independent GGUF parts (not numbered splits) by role:
//! `trunk` (device copy) and `embedding` (BF16 token rows read from SSD) are
//! required; the `mtp` drafter joins the device inventory only when
//! speculative decoding asks for it. `files.json` hashes every file in the
//! directory, release.json and the source `config.json` included, so an
//! artifact proves it was converted from the checkpoint it names.
//!
//! The embedding part is placed last so native finds the token table as the
//! inventory's last tensor. Native maps the returned page-aligned shard plan;
//! offsets stay file-backed.
//!
//! Deviation, named: a part carries no `source_revision`, so it binds by its
//! SHA-256 and by the config in files.json rather than by an upstream commit.

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
    fail, q8_scale_finite, Result, ALIGN, Q4_BYTES, Q4_K, Q8_0, Q8_BYTES, RECIPE, RECIPE_R8,
    VARIANT,
};
use crate::gguf::{GgufError, GgufFile};
use crate::mapped::page_size;
use crate::tensors::{ShardPlan, TensorError, TensorInfo, TensorInventory};

const MAX_METADATA: u64 = 64 * 1024 * 1024;
const RELEASE: &str = "release.json";
const FILES: &str = "files.json";
/// Format version 1 is the plain 13-bit artifact; version 2 is R13+8 and
/// carries its own recipe id, so the two cannot be mistaken for each other.
const FORMAT_VERSION: u64 = 1;
const RESIDUAL_FORMAT_VERSION: u64 = 2;
const CONTEXT: u64 = 262_144;
/// A Q4_K header has two f16 scales; retain an incomplete four-byte header.
const SCALE_TAIL: usize = 3;

/// Device-inventory features: the MTP drafter joins the trunk on request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gemma4VqFeature {
    Trunk,
    Mtp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Role {
    Trunk,
    Embedding,
    Mtp,
}

impl Role {
    fn parse(entry: &Value) -> Result<Self> {
        let (role, residency, required) = match text(entry, "role")? {
            "trunk" => (Self::Trunk, "resident", true),
            "embedding" => (Self::Embedding, "ssd-rows", true),
            "mtp" => (Self::Mtp, "resident-when-enabled", false),
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
            Self::Mtp => b"mtp",
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

fn check_file(bytes: &[u8], entry: &Value) -> Result<()> {
    if number(entry, "bytes")? != bytes.len() as u64 {
        return Err(fail("release file size mismatch"));
    }
    check_hash(bytes, text(entry, "sha256")?)
}

// files.json may list files below subdirectories; parts stay flat names.
fn relative(name: &str) -> Result<()> {
    if name.contains('\\')
        || name.is_empty()
        || !Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
    {
        return Err(fail("release path must be relative"));
    }
    Ok(())
}

fn file_index(root: &Path) -> Result<HashMap<String, Value>> {
    let files = json(&json_bytes(&root.join(FILES))?)?;
    let mut index = HashMap::new();
    for entry in array(&files, "files")? {
        let name = text(entry, "path")?;
        relative(name)?;
        if index.insert(name.to_owned(), entry.clone()).is_some() {
            return Err(fail("duplicate release file"));
        }
    }
    Ok(index)
}

pub(super) fn is_release(root: &Path) -> bool {
    let Ok(bytes) = json_bytes(&root.join(RELEASE)) else {
        return false;
    };
    json(&bytes).is_ok_and(|v| v.get("format").and_then(Value::as_str) == Some("gemma4-a4b-vq8"))
}

/// A file whose hash files.json records, checked against the bytes.
pub(super) fn asset(root: &Path, name: &str) -> Result<Vec<u8>> {
    filename(name)?;
    let index = file_index(root)?;
    let entry = index.get(name).ok_or(fail("missing release asset"))?;
    let bytes = json_bytes(&root.join(name))?;
    check_file(&bytes, entry)?;
    Ok(bytes)
}

struct Part {
    name: String,
    role: Role,
    entry: Value,
}

/// The release version decides which recipe id must accompany it.
fn version_recipe(release: &serde_json::Value) -> Result<bool> {
    let recipe = text(release, "recipe_id")?;
    Ok(match number(release, "format_version")? {
        FORMAT_VERSION => recipe.as_bytes() == RECIPE,
        RESIDUAL_FORMAT_VERSION => recipe.as_bytes() == RECIPE_R8,
        _ => false,
    })
}

/// The verified release entrypoint: the device-copied parts in release order
/// (the trunk, and the MTP drafter on request) with the embedding part last.
fn release_parts(root: &Path, feature: Gemma4VqFeature) -> Result<Vec<Part>> {
    let index = file_index(root)?;
    let bytes = json_bytes(&root.join(RELEASE))?;
    check_file(
        &bytes,
        index.get(RELEASE).ok_or(fail("unlisted release.json"))?,
    )?;
    let release = json(&bytes)?;
    if text(&release, "format")?.as_bytes() != VARIANT
        || text(&release, "variant")?.as_bytes() != VARIANT
        || text(&release, "architecture")? != "gemma4_vq"
        || text(&release, "entrypoint")? != RELEASE
        || text(&release, "source_model_type")? != "gemma4"
        || !version_recipe(&release)?
        || number(&release, "context_length")? != CONTEXT
        || release.get("test_subset").and_then(Value::as_bool) != Some(false)
        || release.get("dry_run").and_then(Value::as_bool) != Some(false)
    {
        return Err(fail("release format/recipe/source model"));
    }
    let mut names = HashSet::new();
    let mut counts = HashMap::new();
    let mut selected = Vec::new();
    let mut embedding = None;
    let mut mtp = None;
    for entry in array(&release, "parts")? {
        let name = text(entry, "path")?;
        filename(name)?;
        let role = Role::parse(entry)?;
        let listed = index
            .get(name)
            .ok_or(fail("part missing from files.json"))?;
        if !name.ends_with(".gguf")
            || !names.insert(name.to_owned())
            || text(entry, "part_recipe_id")? != text(&release, "recipe_id")?
            || listed.get("sha256") != entry.get("sha256")
            || listed.get("bytes") != entry.get("bytes")
        {
            return Err(fail("part identity/role"));
        }
        *counts.entry(role).or_insert(0u32) += 1;
        let part = Part {
            name: name.to_owned(),
            role,
            entry: entry.clone(),
        };
        match role {
            Role::Trunk => selected.push(part),
            Role::Embedding => embedding = Some(part),
            Role::Mtp => mtp = Some(part),
        }
    }
    // One trunk, one embedding table, at most one drafter.
    if counts.get(&Role::Trunk) != Some(&1)
        || counts.get(&Role::Embedding) != Some(&1)
        || counts.get(&Role::Mtp).copied().unwrap_or(0) > 1
        || (feature == Gemma4VqFeature::Mtp && mtp.is_none())
    {
        return Err(fail("missing/duplicate release parts"));
    }
    if feature == Gemma4VqFeature::Mtp {
        selected.extend(mtp);
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
    Ok(selected)
}

fn parts(root: &Path, feature: Gemma4VqFeature) -> Result<Vec<Part>> {
    release_parts(root, feature)
}

pub(super) fn first_part(root: &Path) -> Result<PathBuf> {
    Ok(root.join(&parts(root, Gemma4VqFeature::Trunk)?[0].name))
}

pub(super) fn part_count(root: &Path) -> Result<u32> {
    Ok(parts(root, Gemma4VqFeature::Trunk)?.len() as u32)
}

/// Every tensor of the trunk parts, from the GGUF directories alone.
///
/// This is the cheap read: `TensorInventory::open` hashes every part
/// (`verify_part`, 9.4 GiB on the 4070 release, 22 s) which a
/// `--check-config` preflight must not do. Names, types, offsets and byte
/// sizes are all the resident-span arithmetic needs, and both live in the
/// tensor directory.
pub(super) fn tensor_directory(root: &Path) -> Result<Vec<TensorInfo>> {
    let mut out = Vec::new();
    for part in parts(root, Gemma4VqFeature::Trunk)? {
        let path = root.join(&part.name);
        let g = GgufFile::open(&path)?;
        out.extend(TensorInventory::from_file(&path, &g)?.tensors);
    }
    Ok(out)
}

/// On-disk bytes of the device-copied parts (trunk, and the MTP drafter with
/// `Gemma4VqFeature::Mtp`). Hashes are checked when the release opens, not here.
pub(super) fn resident_bytes(root: &Path, feature: Gemma4VqFeature) -> Result<u64> {
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
static VERIFIED: Mutex<Vec<(PathBuf, Gemma4VqFeature, Stamp, TensorInventory)>> =
    Mutex::new(Vec::new());

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

pub(super) fn open_dir(root: &Path, feature: Gemma4VqFeature) -> Result<TensorInventory> {
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
    if g.get_string("gemma4_vq.part_role") != Some(part.role.metadata())
        || g.get_string("gemma4_vq.source_revision")
            != Some(text(&part.entry, "source_revision")?.as_bytes())
    {
        return Err(fail("part role/source revision mismatch"));
    }
    let inv = super::streamed(|| TensorInventory::from_file(&path, &g))?;
    if inv.shards.len() != 1 || inv.alignment != ALIGN {
        return Err(fail("part is not one aligned GGUF"));
    }
    // The embedding part holds the token table alone.
    let embedding = inv.tensors.len() == 1 && inv.tensors[0].name == super::EMBEDDING;
    if (part.role == Role::Embedding) != embedding
        || inv.tensors.iter().any(|t| t.name == super::EMBEDDING) != embedding
        || number(&part.entry, "tensors")? != inv.tensors.len() as u64
    {
        return Err(fail("part tensors/embedding table"));
    }
    if number(&part.entry, "bytes")? != g.as_bytes().len() as u64 {
        return Err(fail("release file size mismatch"));
    }
    let quant: Vec<(u64, u64, u64)> = inv
        .tensors
        .iter()
        .filter_map(|t| match t.typ {
            Q8_0 => Some((t.abs_offset, t.bytes, Q8_BYTES)),
            Q4_K => Some((t.abs_offset, t.bytes, Q4_BYTES)),
            _ => None,
        })
        .collect();
    let digest = part_digest(&path, g.as_bytes(), Scan::new(&quant))?;
    check_digest(&digest, text(&part.entry, "sha256")?)?;
    let size = g.as_bytes().len() as u64;
    drop(g);
    drop_cache(&path);
    Ok((path, size, inv))
}

/// Parts hash through O_DIRECT reads: streaming ~92 GB through the page cache
/// fragments the memory the native device pages come from. The mapped bytes
/// are the fallback. The same pass checks the quant scales `Scan` covers.
fn part_digest(path: &Path, mapped: &[u8], mut scan: Scan) -> Result<Vec<u8>> {
    #[cfg(target_os = "linux")]
    if let Some(digest) = direct_digest(path, &mut scan)? {
        return Ok(digest);
    }
    scan.feed(0, mapped)?;
    Ok(Sha256::digest(mapped).to_vec())
}

/// Quant scale checks as a part streams by (spans in part offsets). A block's
/// scale bytes may straddle chunks; retain the last three bytes for Q4_K's
/// two f16 scales, as the Ling-3.1 scanner does.
struct Scan<'a> {
    blocks: &'a [(u64, u64, u64)],
    tail: [u8; SCALE_TAIL],
}

impl<'a> Scan<'a> {
    fn new(blocks: &'a [(u64, u64, u64)]) -> Self {
        Self {
            blocks,
            tail: [0; SCALE_TAIL],
        }
    }

    fn feed(&mut self, pos: u64, chunk: &[u8]) -> Result<()> {
        let end = pos + chunk.len() as u64;
        // A block waits until both scales fit; earlier bytes stay in tail.
        let from = pos.saturating_sub(SCALE_TAIL as u64);
        let tail = self.tail;
        let byte = |at: u64| {
            if at >= pos {
                chunk[(at - pos) as usize]
            } else {
                tail[SCALE_TAIL - (pos - at) as usize]
            }
        };
        for &(start, bytes, block) in self.blocks {
            let stop = start + bytes;
            if stop <= from || start >= end {
                continue;
            }
            let mut at = start + from.saturating_sub(start).div_ceil(block) * block;
            let scale_bytes = if block == Q4_BYTES { 4 } else { 2 };
            while at < stop && at + scale_bytes <= end {
                let scale = [byte(at), byte(at + 1)];
                if !q8_scale_finite(&scale) {
                    return Err(fail("nonfinite quant scale"));
                }
                if block == Q4_BYTES {
                    let min = [byte(at + 2), byte(at + 3)];
                    if !q8_scale_finite(&min) {
                        return Err(fail("nonfinite quant scale"));
                    }
                }
                at += block;
            }
        }
        let take = chunk.len().min(SCALE_TAIL);
        self.tail.copy_within(take.., 0);
        self.tail[SCALE_TAIL - take..].copy_from_slice(&chunk[chunk.len() - take..]);
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

fn verify_dir(root: &Path, feature: Gemma4VqFeature) -> Result<TensorInventory> {
    let parts = parts(root, feature)?;
    let page = page_size();
    let mut result = TensorInventory {
        shards: Vec::new(),
        tensors: Vec::new(),
        data_pos: 0,
        alignment: ALIGN,
        page,
    };
    // Parts hash independently; workers pull parts in release order and
    // assembly below stays sequential, so offsets and the first reported
    // error match a serial pass.
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
            for (i, part) in handle.join().expect("Gemma-4 VQ part verifier panicked") {
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

    fn scan(data: &[u8], blocks: &[(u64, u64, u64)], chunk: usize) -> Result<()> {
        let mut scan = Scan::new(blocks);
        for (i, part) in data.chunks(chunk).enumerate() {
            scan.feed((i * chunk) as u64, part)?;
        }
        Ok(())
    }

    // A NaN/inf scale anywhere in a quant span must be found under any
    // chunking, including scales that straddle a chunk boundary.
    #[test]
    fn scan_finds_straddling_scales() {
        let q8 = [(1024, 3 * Q8_BYTES, Q8_BYTES)];
        let clean = vec![0u8; 1536];
        let last = 1024 + 2 * Q8_BYTES as usize;
        for byte in [last + 1, 1024 + 1] {
            let mut bad = clean.clone();
            bad[byte] = 0x7c;
            for chunk in [1, 2, 3, 7, 34, 144, last + 1, 1536] {
                assert!(scan(&clean, &q8, chunk).is_ok(), "chunk {chunk}");
                assert!(scan(&bad, &q8, chunk).is_err(), "byte {byte} chunk {chunk}");
            }
        }
        // The Q4_K minimum scale sits two bytes after d; a NaN there counts.
        let q4 = [(1024, 2 * Q4_BYTES, Q4_BYTES)];
        for byte in [1024 + 1, 1024 + 3, 1024 + Q4_BYTES as usize + 3] {
            let mut bad = clean.clone();
            bad[byte] = 0x7c;
            for chunk in [1, 2, 3, 7, 144, 1025, 1026, 1027, 1536] {
                assert!(scan(&clean, &q4, chunk).is_ok(), "chunk {chunk}");
                assert!(scan(&bad, &q4, chunk).is_err(), "byte {byte} chunk {chunk}");
            }
        }
    }

    #[test]
    fn roles_carry_their_residency() {
        let trunk = json!({"role": "trunk", "residency": "resident", "required": true});
        let embedding = json!({"role": "embedding", "residency": "ssd-rows", "required": true});
        let mtp = json!({"role": "mtp", "residency": "resident-when-enabled", "required": false});
        assert_eq!(Role::parse(&trunk).unwrap(), Role::Trunk);
        assert_eq!(Role::parse(&embedding).unwrap(), Role::Embedding);
        assert_eq!(Role::parse(&mtp).unwrap(), Role::Mtp);
        let mut wrong = mtp.clone();
        wrong["required"] = json!(true);
        assert!(Role::parse(&wrong).is_err());
        assert!(Role::parse(
            &json!({"role": "vision", "residency": "resident", "required": false})
        )
        .is_err());
    }

    #[test]
    fn paths_stay_inside_the_release() {
        assert!(filename("gemma4-vq8-trunk.gguf").is_ok());
        for bad in ["../x.gguf", "a/b.gguf", "a\\b.gguf", ""] {
            assert!(filename(bad).is_err(), "{bad}");
        }
    }

    /// Binds the reader to a real release directory when the operator offers
    /// one: `GEMMA4_VQ_RELEASE=<dir> cargo test -p ds4-core --lib gemma4`.
    /// Identity, files.json, both part hashes, the GVQ1 headers, the book
    /// links and the layout all have to agree for this to pass.
    #[test]
    fn a_release_directory_verifies() {
        let Ok(dir) = std::env::var("GEMMA4_VQ_RELEASE") else {
            return;
        };
        let root = Path::new(&dir);
        let id = crate::gemma4_vq::model::identify(root).expect("identify");
        assert_eq!(id.shape.variant, crate::Variant::Gemma4A4bVq);
        assert_eq!(id.split_count, 2);
        let inventory = open_dir(root, Gemma4VqFeature::Trunk).expect("open the release");
        crate::gemma4_vq::validate_inventory(&inventory, Gemma4VqFeature::Trunk).expect("layout");
        assert_eq!(inventory.tensors.len(), 12_177);
        assert_eq!(inventory.shards.len(), 2);
        // The embedding part is last, so native finds the token table there.
        assert_eq!(
            inventory.tensors.last().unwrap().name,
            crate::gemma4_vq::EMBEDDING
        );
        // The drafter, when this release ships it: 48 more tensors, its own
        // layout, and the trunk part still first.
        if let Ok(with_mtp) = open_dir(root, Gemma4VqFeature::Mtp) {
            crate::gemma4_vq::validate_inventory(&with_mtp, Gemma4VqFeature::Mtp)
                .expect("mtp layout");
            assert_eq!(with_mtp.tensors.len(), 12_177 + 48);
            assert_eq!(with_mtp.shards.len(), 3);
        }
    }
}
