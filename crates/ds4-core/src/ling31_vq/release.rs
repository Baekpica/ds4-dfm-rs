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

use super::{
    fail, fp8_finite, part_recipe_ok, q4_scales_finite, q8_scale_finite, Result, ALIGN,
    FORMAT_VERSION, FP8, Q4_BYTES, Q4_K, Q8_0, Q8_BYTES, VARIANT,
};
use crate::gguf::{GgufError, GgufFile};
use crate::mapped::page_size;
use crate::tensors::{ShardPlan, TensorError, TensorInventory};

const MAX_METADATA: u64 = 64 * 1024 * 1024;
const RELEASE: &str = "release.json";
const FILES: &str = "files.json";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Trunk,
    Mtp,
    Embedding,
}

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
    role: Role,
    entry: Value,
    recipe: String,
    source: String,
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
        || !part_recipe_ok(
            text(&release, "recipe_id")?.as_bytes(),
            text(&release, "recipe_id")?.as_bytes(),
        )
        || Some(text(&release, "source_revision")?)
            != super::source_revision(text(&release, "recipe_id")?.as_bytes())
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
            "trunk" => Role::Trunk,
            "mtp" => Role::Mtp,
            "embedding" if text(&release, "recipe_id")?.as_bytes() == super::RECIPE_1019 => {
                Role::Embedding
            }
            _ => return Err(fail("unknown part role")),
        };
        let listed = index
            .get(name)
            .ok_or(fail("part missing from files.json"))?;
        if !name.ends_with(".gguf")
            || !names.insert(name.to_owned())
            || entry.get("required").and_then(Value::as_bool) != Some(role != Role::Mtp)
            || listed.get("sha256") != entry.get("sha256")
            || listed.get("bytes") != entry.get("bytes")
        {
            return Err(fail("part identity/role"));
        }
        trunk |= role == Role::Trunk;
        mtp |= role == Role::Mtp;
        if role != Role::Mtp || feature == Feature::Mtp {
            selected.push(Part {
                name: name.to_owned(),
                role,
                entry: entry.clone(),
                recipe: text(&release, "recipe_id")?.to_owned(),
                source: text(&release, "source_revision")?.to_owned(),
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
    if text(&release, "recipe_id")?.as_bytes() == super::RECIPE_1019 {
        if selected
            .iter()
            .filter(|p| p.role == Role::Embedding)
            .count()
            != 1
        {
            return Err(fail("need exactly one embedding part"));
        }
        // Keep the host-only table last, so it has no GPU pointer record.
        selected.sort_by_key(|p| p.role == Role::Embedding);
    }
    Ok(selected)
}

pub(super) fn first_part(root: &Path) -> Result<PathBuf> {
    Ok(root.join(&parts(root, Feature::Trunk)?[0].name))
}

pub(super) fn part_count(root: &Path, feature: Feature) -> Result<u32> {
    Ok(parts(root, feature)?.len() as u32)
}

/// Device-resident bytes of the selected parts, excluding the 1019 SSD
/// embedding. Hashes are checked when the release opens, not here.
pub(super) fn part_bytes(root: &Path, feature: Feature) -> Result<u64> {
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
    let role: &[u8] = match part.role {
        Role::Trunk => b"trunk",
        Role::Mtp => b"mtp",
        Role::Embedding => b"embedding",
    };
    if g.get_string("ling_vq.part_role") != Some(role)
        || g.get_string("ling_vq.source_revision") != Some(part.source.as_bytes())
        || !part_recipe_ok(
            part.recipe.as_bytes(),
            g.get_string("ling_vq.recipe_id").unwrap_or_default(),
        )
    {
        return Err(fail("part role/source revision/recipe mismatch"));
    }
    let inv = super::streamed(|| TensorInventory::from_file(&path, &g))?;
    if part.recipe.as_bytes() == super::RECIPE_1019 {
        let embedding = inv.tensors.len() == 1 && inv.tensors[0].name == super::EMBEDDING;
        if (part.role == Role::Embedding) != embedding
            || inv.tensors.iter().any(|t| t.name == super::EMBEDDING) != embedding
        {
            return Err(fail("embedding tensor outside its part"));
        }
        let recipe = json(&asset(root, "recipe.json")?)?;
        if text(&recipe, "recipe_id")? != part.recipe
            || text(&recipe, "source_revision")? != part.source
        {
            return Err(fail("recipe identity mismatch"));
        }
        let allocation = recipe
            .get("layer_k")
            .and_then(Value::as_object)
            .ok_or(fail("missing precision allocation"))?;
        if allocation.len() != 69
            || (4..73).any(|i| allocation.get(&i.to_string()).and_then(Value::as_u64) != Some(4096))
        {
            return Err(fail("invalid 1019 precision allocation"));
        }
        for tensor in inv.tensors.iter().filter(|t| t.typ == super::CODEBOOK) {
            let layer = tensor
                .name
                .strip_prefix("model.layers.")
                .and_then(|n| n.strip_suffix(".mlp.vq_codebook"))
                .ok_or(fail("codebook name"))?;
            if allocation.get(layer).and_then(Value::as_u64) != Some(tensor.dim[1]) {
                return Err(fail("stored codebook differs from selected allocation"));
            }
        }
    }
    if inv.shards.len() != 1 || inv.alignment != ALIGN {
        return Err(fail("part is not one aligned GGUF"));
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
    let (q4, q8, fp8) = (spans(Q4_K), spans(Q8_0), spans(FP8));
    let digest = part_digest(&path, g.as_bytes(), Scan::new(&q4, &q8, &fp8))?;
    check_digest(&digest, text(&part.entry, "sha256")?)?;
    let size = g.as_bytes().len() as u64;
    drop(g);
    drop_cache(&path);
    Ok((path, size, inv))
}

/// Parts hash through O_DIRECT reads: streaming ~110 GiB through the page
/// cache fragments the free memory the native 2 MiB device pages come from
/// (GB10 UMA). The mapped bytes are the fallback.
/// The same pass runs the whole-tensor content checks of `scan`.
fn part_digest(path: &Path, mapped: &[u8], mut scan: Scan) -> Result<Vec<u8>> {
    #[cfg(target_os = "linux")]
    if let Some(digest) = direct_digest(path, &mut scan)? {
        return Ok(digest);
    }
    scan.feed(0, mapped)?;
    Ok(Sha256::digest(mapped).to_vec())
}

/// Content checks run as a part streams by (spans in part offsets): every
/// Q4_K and Q8_0 block's f16 scales and every FP8 byte. A block's scale
/// bytes may straddle two chunks; the previous chunk's last three bytes carry.
struct Scan<'a> {
    q4: &'a [(u64, u64)],
    q8: &'a [(u64, u64)],
    fp8: &'a [(u64, u64)],
    tail: [u8; 3],
}

impl<'a> Scan<'a> {
    fn new(q4: &'a [(u64, u64)], q8: &'a [(u64, u64)], fp8: &'a [(u64, u64)]) -> Self {
        Self {
            q4,
            q8,
            fp8,
            tail: [0; 3],
        }
    }

    fn feed(&mut self, pos: u64, chunk: &[u8]) -> Result<()> {
        let end = pos + chunk.len() as u64;
        for &(start, bytes) in self.fp8 {
            let (from, to) = (start.max(pos), (start + bytes).min(end));
            if from < to && !fp8_finite(&chunk[(from - pos) as usize..(to - pos) as usize]) {
                return Err(fail("nonfinite FP8 weight"));
            }
        }
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

/// A page-aligned O_DIRECT buffer that returns its pages on drop. A freed
/// 16 MiB Vec raises glibc's mmap threshold, so later ones stay in the
/// verifier threads' arenas: ~0.5 GiB of a loaded server's RSS.
#[cfg(target_os = "linux")]
struct DirectBuf(*mut u8, usize);

#[cfg(target_os = "linux")]
impl DirectBuf {
    fn new(len: usize) -> Option<Self> {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        (p != libc::MAP_FAILED).then_some(Self(p.cast(), len))
    }

    fn slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0, self.1) }
    }
}

#[cfg(target_os = "linux")]
impl Drop for DirectBuf {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.0.cast(), self.1) };
    }
}

#[cfg(target_os = "linux")]
fn direct_digest(path: &Path, scan: &mut Scan) -> Result<Option<Vec<u8>>> {
    use std::os::unix::fs::OpenOptionsExt;
    const DIRECT_CHUNK: usize = 16 << 20;
    let Ok(mut file) = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    else {
        return Ok(None);
    };
    let Some(mut raw) = DirectBuf::new(DIRECT_CHUNK) else {
        return Ok(None);
    };
    let buf = raw.slice();
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

#[cfg(test)]
mod tests {
    use super::*;

    // Two Q4_K spans of three blocks; any chunking must find a NaN scale,
    // including one whose four bytes straddle a chunk boundary.
    fn scan(data: &[u8], spans: &[(u64, u64)], chunk: usize) -> Result<()> {
        scan_fp8(data, spans, &[], chunk)
    }

    fn scan_fp8(data: &[u8], q4: &[(u64, u64)], fp8: &[(u64, u64)], chunk: usize) -> Result<()> {
        scan_q8(data, q4, &[], fp8, chunk)
    }

    fn scan_q8(
        data: &[u8],
        q4: &[(u64, u64)],
        q8: &[(u64, u64)],
        fp8: &[(u64, u64)],
        chunk: usize,
    ) -> Result<()> {
        let mut scan = Scan::new(q4, q8, fp8);
        for (i, part) in data.chunks(chunk).enumerate() {
            scan.feed((i * chunk) as u64, part)?;
        }
        Ok(())
    }

    // A Q8_0 block's f16 scale may straddle a chunk boundary too.
    #[test]
    fn q8_scan_finds_straddling_scales() {
        let spans = [(128, 3 * Q8_BYTES)];
        let clean = vec![0u8; 512];
        let at = 128 + 2 * Q8_BYTES as usize;
        let mut bad = clean.clone();
        bad[at + 1] = 0x7c; // +inf d of the last block
        for chunk in [1, 2, 3, 34, at + 1, 512] {
            assert!(
                scan_q8(&clean, &[], &spans, &[], chunk).is_ok(),
                "chunk {chunk}"
            );
            assert!(
                scan_q8(&bad, &[], &spans, &[], chunk).is_err(),
                "chunk {chunk}"
            );
        }
    }

    #[test]
    fn q4_scan_finds_straddling_scales() {
        let spans = [(128, 3 * Q4_BYTES), (1024, 3 * Q4_BYTES)];
        let clean = vec![0u8; 1536];
        let at = 1024 + 2 * Q4_BYTES as usize;
        // +inf d (high byte at +1) or dmin (+3) of the last block.
        for byte in [at + 1, at + 3] {
            let mut bad = clean.clone();
            bad[byte] = 0x7c;
            for chunk in [1, 2, 3, 7, 144, at + 2, 1536] {
                assert!(scan(&clean, &spans, chunk).is_ok(), "chunk {chunk}");
                assert!(
                    scan(&bad, &spans, chunk).is_err(),
                    "byte {byte} chunk {chunk}"
                );
            }
        }
    }

    #[test]
    fn fp8_scan_finds_nan_bytes() {
        let fp8 = [(64, 50)];
        let clean = vec![0x7eu8; 256];
        for chunk in [1, 7, 33, 256] {
            assert!(scan_fp8(&clean, &[], &fp8, chunk).is_ok(), "chunk {chunk}");
            let mut bad = clean.clone();
            bad[113] = 0xff; // last byte of the span: E4M3FN NaN
            assert!(scan_fp8(&bad, &[], &fp8, chunk).is_err(), "chunk {chunk}");
            bad[113] = 0x7e;
            bad[114] = 0x7f; // outside the span
            assert!(scan_fp8(&bad, &[], &fp8, chunk).is_ok(), "chunk {chunk}");
        }
    }
}
