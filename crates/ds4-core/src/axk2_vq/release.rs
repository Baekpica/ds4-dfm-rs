//! `release.json` lists independent GGUF parts (not numbered splits), every
//! one required: the trunk parts the device copy holds and the one
//! `embedding` part (the BF16 token table read in rows from SSD). Native
//! maps the returned page-aligned shard plan; offsets stay file-backed.

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
    fail, q4_scales_finite, q8_scale_finite, Result, ALIGN, FORMAT_VERSION, Q4_BYTES, Q4_K, Q8_0,
    Q8_BYTES, RECIPE, VARIANT,
};
use crate::gguf::{GgufError, GgufFile};
use crate::mapped::page_size;
use crate::tensors::{ShardPlan, TensorError, TensorInventory};

const MAX_METADATA: u64 = 64 * 1024 * 1024;
const RELEASE: &str = "release.json";
const FILES: &str = "files.json";
/// Pinned skt/A.X-K2 revision of every stored weight.
const SOURCE_REVISION: &str = "d0805cfd1fc647b4078c0d9f232c80d34ee0c893";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Trunk,
    Embedding,
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

// files.json also lists files below subdirectories (assets/, tools/);
// those stay relative, while parts and assets are flat names.
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
    json(&bytes).is_ok_and(|v| v.get("format").and_then(Value::as_str) == Some("axk2-vq8"))
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
    role: Role,
    entry: Value,
}

/// The verified release entrypoint and its parts in manifest order (expert
/// layers, body, then the embedding part, as published).
fn parts(root: &Path) -> Result<Vec<Part>> {
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
    for entry in array(&release, "parts")? {
        let name = text(entry, "path")?;
        filename(name)?;
        let role = match text(entry, "role")? {
            "trunk" => Role::Trunk,
            "embedding" => Role::Embedding,
            _ => return Err(fail("unknown part role")),
        };
        let listed = index
            .get(name)
            .ok_or(fail("part missing from files.json"))?;
        if !name.ends_with(".gguf")
            || !names.insert(name.to_owned())
            || entry.get("required").and_then(Value::as_bool) != Some(true)
            || listed.get("sha256") != entry.get("sha256")
            || listed.get("bytes") != entry.get("bytes")
        {
            return Err(fail("part identity/role"));
        }
        selected.push(Part {
            name: name.to_owned(),
            role,
            entry: entry.clone(),
        });
    }
    let embedding = selected
        .iter()
        .filter(|p| p.role == Role::Embedding)
        .count();
    if embedding != 1 || selected.len() < 2 {
        return Err(fail("need trunk parts and exactly one embedding part"));
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
    Ok(root.join(&parts(root)?[0].name))
}

pub(super) fn part_count(root: &Path) -> Result<u32> {
    Ok(parts(root)?.len() as u32)
}

/// On-disk bytes of the trunk parts: what the device copy holds. Hashes are
/// checked when the release opens, not here.
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

/// File mtimes are coarse (one jiffy); only entries this old are cached, so
/// any later rewrite is guaranteed a different mtime.
const SETTLED: Duration = Duration::from_secs(2);
const MAX_VERIFIERS: usize = 16;

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
    let role: &[u8] = match part.role {
        Role::Trunk => b"trunk",
        Role::Embedding => b"embedding",
    };
    if g.get_string("axk2_vq.part_role") != Some(role)
        || g.get_string("axk2_vq.source_revision") != Some(SOURCE_REVISION.as_bytes())
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
    {
        return Err(fail("embedding tensor outside its part"));
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

/// Parts hash through O_DIRECT reads: streaming ~100 GB through the page
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
    // Parts hash independently (~100 GB); workers pull parts in release
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
            for (i, part) in handle.join().expect("A.X-K2 VQ part verifier panicked") {
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

    fn scan(data: &[u8], q4: &[(u64, u64)], q8: &[(u64, u64)], chunk: usize) -> Result<()> {
        let mut scan = Scan::new(q4, q8);
        for (i, part) in data.chunks(chunk).enumerate() {
            scan.feed((i * chunk) as u64, part)?;
        }
        Ok(())
    }

    // A NaN/inf scale anywhere in a Q4_K or Q8_0 span must be found under
    // any chunking, including scales that straddle a chunk boundary.
    #[test]
    fn scan_finds_straddling_scales() {
        let q4 = [(128, 3 * Q4_BYTES)];
        let q8 = [(1024, 3 * Q8_BYTES)];
        let clean = vec![0u8; 1536];
        let last_q4 = 128 + 2 * Q4_BYTES as usize;
        let last_q8 = 1024 + 2 * Q8_BYTES as usize;
        // +inf d (high byte at +1) or dmin (+3) of the last Q4_K block, the
        // last Q8_0 block's d.
        for byte in [last_q4 + 1, last_q4 + 3, last_q8 + 1] {
            let mut bad = clean.clone();
            bad[byte] = 0x7c;
            for chunk in [1, 2, 3, 7, 34, 144, last_q8 + 1, 1536] {
                assert!(scan(&clean, &q4, &q8, chunk).is_ok(), "chunk {chunk}");
                assert!(
                    scan(&bad, &q4, &q8, chunk).is_err(),
                    "byte {byte} chunk {chunk}"
                );
            }
        }
    }
}
