//! `release.json` selects independent GGUF parts (not numbered splits) by
//! feature before anything maps them: trunk always, `mtp` only on request,
//! so MTP-off serving maps no layer-72 bytes. Native maps the returned
//! page-aligned shard plan; offsets stay file-backed.

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

use super::{fail, Result, ALIGN, FORMAT_VERSION, RECIPE, VARIANT};
use crate::gguf::{GgufError, GgufFile};
use crate::mapped::page_size;
use crate::tensors::{ShardPlan, TensorError, TensorInventory};

const MAX_METADATA: u64 = 64 * 1024 * 1024;
const RELEASE: &str = "release.json";
const FILES: &str = "files.json";
/// Pinned inclusionAI/Ling-3.1-flash revision of every stored weight.
const SOURCE_REVISION: &str = "a4ccd6b45c002ddb236f9cb77d52cc427482bf19";

/// `release.json` features. MTP adds the role=mtp parts to the trunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Feature {
    Trunk,
    Mtp,
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
    if expected.len() != 64 || !expected.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(fail("invalid release SHA256"));
    }
    let digest = Sha256::digest(bytes);
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

// files.json also lists documentation below subdirectories (licenses/,
// source/); those stay relative, while parts and assets are flat names.
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
    json(&bytes).is_ok_and(|v| v.get("format").and_then(Value::as_str) == Some("ling31-vq8"))
}

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
    role: Feature,
    entry: Value,
}

/// The verified release entrypoint and the parts the feature selects, in
/// manifest order (trunk first, as published).
fn parts(root: &Path, feature: Feature) -> Result<Vec<Part>> {
    let index = file_index(root)?;
    let bytes = json_bytes(&root.join(RELEASE))?;
    check_file(
        &bytes,
        index.get(RELEASE).ok_or(fail("unlisted release.json"))?,
    )?;
    let release = json(&bytes)?;
    if text(&release, "format")?.as_bytes() != VARIANT
        || number(&release, "format_version")? != u64::from(FORMAT_VERSION)
        || text(&release, "recipe_id")?.as_bytes() != RECIPE
        || text(&release, "source_revision")? != SOURCE_REVISION
        || text(&release, "entrypoint")? != RELEASE
    {
        return Err(fail("release format/recipe/source revision"));
    }
    let mut names = HashSet::new();
    let mut selected = Vec::new();
    let (mut trunk, mut mtp) = (false, false);
    for entry in array(&release, "parts")? {
        let name = text(entry, "path")?;
        filename(name)?;
        let role = match text(entry, "role")? {
            "trunk" => Feature::Trunk,
            "mtp" => Feature::Mtp,
            _ => return Err(fail("unknown part role")),
        };
        let listed = index
            .get(name)
            .ok_or(fail("part missing from files.json"))?;
        if !name.ends_with(".gguf")
            || !names.insert(name.to_owned())
            || entry.get("required").and_then(Value::as_bool) != Some(role == Feature::Trunk)
            || listed.get("sha256") != entry.get("sha256")
            || listed.get("bytes") != entry.get("bytes")
        {
            return Err(fail("part identity/role"));
        }
        trunk |= role == Feature::Trunk;
        mtp |= role == Feature::Mtp;
        if role == Feature::Trunk || feature == Feature::Mtp {
            selected.push(Part {
                name: name.to_owned(),
                role,
                entry: entry.clone(),
            });
        }
    }
    if !trunk || (feature == Feature::Mtp && !mtp) {
        return Err(fail("missing release feature parts"));
    }
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

pub(super) fn first_part(root: &Path) -> Result<PathBuf> {
    Ok(root.join(&parts(root, Feature::Trunk)?[0].name))
}

pub(super) fn part_count(root: &Path, feature: Feature) -> Result<u32> {
    Ok(parts(root, feature)?.len() as u32)
}

/// On-disk bytes of the selected parts: the span the model maps. Hashes are
/// checked when the release opens, not here.
pub(super) fn part_bytes(root: &Path, feature: Feature) -> Result<u64> {
    parts(root, feature)?.iter().try_fold(0u64, |total, part| {
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
    let role: &[u8] = if part.role == Feature::Trunk {
        b"trunk"
    } else {
        b"mtp"
    };
    if g.get_string("ling_vq.part_role") != Some(role)
        || g.get_string("ling_vq.source_revision") != Some(SOURCE_REVISION.as_bytes())
    {
        return Err(fail("part role/source revision mismatch"));
    }
    let inv = TensorInventory::from_file(&path, &g)?;
    if inv.shards.len() != 1 || inv.alignment != ALIGN {
        return Err(fail("part is not one aligned GGUF"));
    }
    check_file(g.as_bytes(), &part.entry)?;
    let size = g.as_bytes().len() as u64;
    drop(g);
    drop_cache(&path);
    Ok((path, size, inv))
}

/// Hashing reads all ~110 GiB. Cached, those pages compete with the native
/// device copy for the same UMA memory (GB10: the load reclaims them under
/// pressure), so return them once a part is verified.
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
    // Parts hash independently (~110 GiB); workers pull parts in release
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
            for (i, part) in handle.join().expect("Ling VQ part verifier panicked") {
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
