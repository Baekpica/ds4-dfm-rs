//! Assemble GVQ1's named parts without changing or concatenating the files.
//! Native maps the returned page-aligned shard plan; offsets stay file-backed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{fail, raw, Result, ALIGN, FORMAT_VERSION, PATCH};
use crate::gguf::{GgufError, GgufFile};
use crate::mapped::page_size;
use crate::tensors::{ShardPlan, TensorError, TensorInfo, TensorInventory};

const MAX_METADATA: u64 = 64 * 1024 * 1024;
const MANIFEST: &str = "tensor-manifest.json";

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
    let digest = Sha256::digest(bytes);
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

pub(super) fn asset(root: &Path, name: &str) -> Result<Vec<u8>> {
    filename(name)?;
    let files = json(&json_bytes(&root.join("files.json"))?)?;
    let matches: Vec<_> = array(&files, "files")?
        .iter()
        .filter(|e| e.get("path").and_then(Value::as_str) == Some(name))
        .collect();
    if matches.len() != 1 {
        return Err(fail("missing/duplicate release asset"));
    }
    let bytes = json_bytes(&root.join(name))?;
    check_file(&bytes, matches[0])?;
    Ok(bytes)
}

// The `.gguf` parts files.json lists, sorted.
fn parts(root: &Path) -> Result<Vec<String>> {
    let files = json(&json_bytes(&root.join("files.json"))?)?;
    let mut parts = Vec::new();
    for entry in array(&files, "files")? {
        let name = text(entry, "path")?;
        filename(name)?;
        if name.ends_with(".gguf") {
            parts.push(name.to_owned());
        }
    }
    parts.sort_unstable();
    if parts.is_empty() || parts.windows(2).any(|p| p[0] == p[1]) {
        return Err(fail("missing/duplicate release parts"));
    }
    Ok(parts)
}

pub(super) fn first_part(root: &Path) -> Result<(std::path::PathBuf, u32)> {
    let parts = parts(root)?;
    Ok((root.join(&parts[0]), parts.len() as u32))
}

/// On-disk bytes of every part: the span the model maps. Hashes are checked
/// when the release opens, not here.
pub(super) fn part_bytes(root: &Path) -> Result<u64> {
    parts(root)?.iter().try_fold(0u64, |total, name| {
        let len = std::fs::metadata(root.join(name))
            .map_err(GgufError::Io)?
            .len();
        Ok(total.saturating_add(len))
    })
}

fn check_tensor(g: &GgufFile, tensor: &TensorInfo, entry: &Value) -> Result<()> {
    let dims: Vec<_> = tensor.dim[..tensor.ndim as usize]
        .iter()
        .rev()
        .copied()
        .map(Value::from)
        .collect();
    if number(entry, "type_id")? != u64::from(tensor.typ)
        || number(entry, "payload_bytes")? != tensor.bytes
        || array(entry, "stored_shape")? != dims
        || entry.get("finite") != Some(&Value::Bool(true))
    {
        return Err(fail("manifest tensor type/shape/size"));
    }
    let shape = array(entry, "shape")?;
    if (tensor.name == PATCH && shape != [1024, 3, 2, 14, 14].map(Value::from))
        || (tensor.name != PATCH && shape != dims)
    {
        return Err(fail("manifest logical shape"));
    }
    check_hash(raw(g.as_bytes(), tensor)?, text(entry, "payload_sha256")?)
}

const MAX_VERIFIERS: usize = 16;
/// File mtimes are coarse (one jiffy); only entries this old are cached, so
/// any later rewrite is guaranteed a different mtime.
const SETTLED: Duration = Duration::from_secs(2);

type Stamp = Vec<(OsString, u64, SystemTime)>;

/// Inventories already verified in this process, keyed by directory and
/// the size/mtime of every entry. Each load path (validation, binding,
/// serving identity) reopens the directory; one verification is enough.
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
fn verify_part(
    root: &Path,
    part: &str,
    expected: &HashMap<&str, &Value>,
    entry: &Value,
    revision: &str,
) -> Result<(PathBuf, u64, TensorInventory)> {
    let path = root.join(part);
    let g = GgufFile::open(&path)?;
    if g.get_string("glm_vq.source_revision") != Some(revision.as_bytes())
        || g.get_u32("glm_vq.format_version") != Some(FORMAT_VERSION)
    {
        return Err(fail("part format/source revision mismatch"));
    }
    let inv = TensorInventory::from_file(&path, &g)?;
    if inv.shards.len() != 1 || inv.tensors.len() != expected.len() {
        return Err(fail("part tensor count"));
    }
    check_file(g.as_bytes(), entry)?;
    for tensor in &inv.tensors {
        let entry = expected
            .get(tensor.name.as_str())
            .ok_or(fail("unexpected part tensor"))?;
        check_tensor(&g, tensor, entry)?;
    }
    Ok((path, g.as_bytes().len() as u64, inv))
}

fn verify_dir(root: &Path) -> Result<TensorInventory> {
    let files = json(&json_bytes(&root.join("files.json"))?)?;
    let mut file_index = HashMap::new();
    for entry in array(&files, "files")? {
        let name = text(entry, "path")?;
        filename(name)?;
        if file_index.insert(name, entry).is_some() {
            return Err(fail("duplicate release file"));
        }
    }
    let bytes = json_bytes(&root.join(MANIFEST))?;
    check_file(
        &bytes,
        file_index
            .get(MANIFEST)
            .ok_or(fail("unlisted tensor manifest"))?,
    )?;
    let manifest = json(&bytes)?;
    let revision = text(&manifest, "source_revision")?;
    if number(&manifest, "format_version")? != u64::from(FORMAT_VERSION)
        || revision.len() != 40
        || !revision.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(fail("release format/source revision"));
    }
    let entries = array(&manifest, "tensors")?;
    if entries.is_empty() || number(&manifest, "output_tensor_count")? != entries.len() as u64 {
        return Err(fail("release tensor count"));
    }
    let mut by_part: BTreeMap<&str, HashMap<&str, &Value>> = BTreeMap::new();
    let mut names = HashSet::new();
    for entry in entries {
        let part = text(entry, "part")?;
        let name = text(entry, "name")?;
        filename(part)?;
        if !part.ends_with(".gguf") || !names.insert(name) {
            return Err(fail("invalid part/duplicate tensor"));
        }
        by_part.entry(part).or_default().insert(name, entry);
    }
    // Neither an omitted manifest part nor an unlisted extra GGUF is a model.
    let declared: HashSet<_> = file_index
        .keys()
        .copied()
        .filter(|p| p.ends_with(".gguf"))
        .collect();
    let wanted: HashSet<_> = by_part.keys().copied().collect();
    if declared != wanted {
        return Err(fail("manifest/file inventory mismatch"));
    }
    let mut actual = HashSet::new();
    for entry in std::fs::read_dir(root).map_err(GgufError::Io)? {
        let entry = entry.map_err(GgufError::Io)?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| fail("non-UTF8 release path"))?;
        if name.ends_with(".gguf") {
            actual.insert(name);
        }
    }
    if actual.len() != wanted.len() || actual.iter().any(|p| !wanted.contains(p.as_str())) {
        return Err(fail("missing/unexpected GGUF part"));
    }
    let page = page_size();
    let mut result = TensorInventory {
        shards: Vec::new(),
        tensors: Vec::new(),
        data_pos: 0,
        alignment: ALIGN,
        page,
    };
    // Parts hash independently (~74 GiB, three digests per byte); workers
    // pull parts in manifest order and assembly below stays sequential, so
    // offsets and the first reported error match a serial pass.
    let parts: Vec<_> = by_part.into_iter().collect();
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
                        let Some((part, expected)) = parts.get(i) else {
                            return done;
                        };
                        let entry = file_index[part];
                        done.push((i, verify_part(root, part, expected, entry, revision)));
                    }
                })
            })
            .collect();
        for handle in handles {
            for (i, part) in handle.join().expect("GLM VQ part verifier panicked") {
                verified[i] = Some(part);
            }
        }
    });
    let mut base = 0u64;
    for part in verified {
        let (path, size, mut inv) = part.expect("every part is verified")?;
        let shard = result.shards.len() as u32;
        if shard == 0 {
            result.data_pos = inv.data_pos;
        }
        for tensor in &mut inv.tensors {
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
