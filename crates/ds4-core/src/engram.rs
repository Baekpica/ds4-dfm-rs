//! DeepSeek V4.1 engram: the rolling n-gram hash and the on-disk table
//! reader.
//!
//! Source of truth is the C engine at `/data/YoungAi` (commit 3946dbc):
//!
//! - `core_v41_engram.c:94-116` `v41_engram_hash`: compressed id via
//!   `token_map`, per-k multipliers, rolling XOR, per-head `% prime + offset`.
//!   The multipliers and primes are read flat in the layer-major order the
//!   engine uses (`mult[ei * G + k]`, `prim[(ei*(G-1) + i-1)*H + h]`), which
//!   is NOT the row-major order their GGUF dims suggest - reading them by
//!   dims silently produces different rows, not an error.
//! - `core_v41_engram.c:24-27` `v41_edio_pread`: an O_DIRECT read of an
//!   arbitrary span through an aligned bounce buffer.
//! - `core_v41_engram.c:29-72` `v41_engram_open_shard`: O_DIRECT first, plain
//!   read plus `FADV_RANDOM` when the filesystem refuses, because 264-byte
//!   random reads through the page cache evict the model mapping and make
//!   temperature-0 runs non-reproducible.
//! - `core_v41_engram.c:236-242` the span check and the two row planes: row r
//!   of layer ei is `head_dim` weight bytes at `weight_off + r*head_dim` and
//!   `head_dim/32` scale bytes at `scale_off + r*(head_dim/32)`.
//!
//! The hash is verified against the engine's own captured rows: the golden
//! set's `p<N>.logits.bin.erows_L01/L14.txt` are the row ids the engine
//! computed for the captured prompts, and `hash_rows` reproduces them
//! (evidence in the port plan, §6.6).

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::ds41::V41Wire;
use crate::gguf::GgufFile;
use crate::tensors::TensorInventory;

/// `V41_EDIO_ALIGN` (`core_v41.h:18`): the O_DIRECT alignment granularity.
pub const EDIO_ALIGN: usize = 4096;

#[derive(Debug)]
pub enum EngramError {
    Open(String),
    Io(String),
    /// The table is shorter than `weight_off + rows*head_dim` (or the scale
    /// plane); the engine refuses the same way (`core_v41_engram.c:238-240`).
    OutOfRange(usize),
    MissingTensor(String),
    TensorType(String),
    ShortTensor(String),
}

impl EngramError {
    pub fn token(&self) -> String {
        match self {
            EngramError::Open(p) => format!("engram-open {p}"),
            EngramError::Io(e) => format!("engram-io {e}"),
            EngramError::OutOfRange(i) => format!("engram-out-of-range {i}"),
            EngramError::MissingTensor(n) => format!("engram-missing-tensor {n}"),
            EngramError::TensorType(n) => format!("engram-tensor-type {n}"),
            EngramError::ShortTensor(n) => format!("engram-short-tensor {n}"),
        }
    }
}

impl std::fmt::Display for EngramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for EngramError {}

/// The four hash constant tensors plus the scalar parameters the hash reads.
/// All four live in the main GGUF and are host-mapped, not uploaded.
pub struct EngramHash {
    pub token_map: Vec<i32>,
    pub multipliers: Vec<i64>,
    pub primes: Vec<i64>,
    pub offsets: Vec<i64>,
    pub max_ngram: u32,
    pub heads: u32,
    pub pad: i32,
    pub n_vocab: u32,
}

/// C `tensor_data` for an i32/i64 tensor: the bytes at the inventory offset.
fn tensor_i64(
    g: &GgufFile,
    inv: &TensorInventory,
    name: &str,
    typ: u32,
) -> Result<Vec<i64>, EngramError> {
    let t = inv
        .find(name)
        .ok_or_else(|| EngramError::MissingTensor(name.into()))?;
    if t.typ != typ {
        return Err(EngramError::TensorType(name.into()));
    }
    let data = g.as_bytes();
    let start = t.abs_offset as usize;
    let end = start + t.bytes as usize;
    let raw = data
        .get(start..end)
        .ok_or_else(|| EngramError::ShortTensor(name.into()))?;
    Ok(raw
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn tensor_i32(g: &GgufFile, inv: &TensorInventory, name: &str) -> Result<Vec<i32>, EngramError> {
    let t = inv
        .find(name)
        .ok_or_else(|| EngramError::MissingTensor(name.into()))?;
    if t.typ != 26 {
        return Err(EngramError::TensorType(name.into()));
    }
    let data = g.as_bytes();
    let start = t.abs_offset as usize;
    let end = start + t.bytes as usize;
    let raw = data
        .get(start..end)
        .ok_or_else(|| EngramError::ShortTensor(name.into()))?;
    Ok(raw
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

impl EngramHash {
    /// The hash constants are required when the wire carries engram layers
    /// (`weights_bind_v41` requires the four names, core_bind_v41.c:47-52).
    pub fn load(
        g: &GgufFile,
        inv: &TensorInventory,
        wire: &V41Wire,
        n_vocab: u32,
    ) -> Result<Option<Self>, EngramError> {
        if wire.engram_layers.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            token_map: tensor_i32(g, inv, "engram.token_map")?,
            multipliers: tensor_i64(g, inv, "engram.multipliers", 27)?,
            primes: tensor_i64(g, inv, "engram.primes", 27)?,
            offsets: tensor_i64(g, inv, "engram.offsets", 27)?,
            max_ngram: wire.engram_max_ngram,
            heads: wire.engram_heads,
            pad: wire.engram_pad as i32,
            n_vocab,
        }))
    }

    /// Columns per position: (max_ngram - 1) heads.
    pub fn cols(&self) -> u32 {
        (self.max_ngram - 1) * self.heads
    }

    /// C `v41_engram_hash` (YoungAi 7fb26cc `core_v41_engram.c:63-90`).
    /// `hist` is the history by absolute position; `p` is the position to hash;
    /// `ei` is the engram index (not the layer number).
    pub fn rows(&self, hist: &[i32], p: i64, ei: u32) -> Vec<i64> {
        self.rows_masked(hist, &[], p, ei)
    }

    /// Vision extension from YoungAi 846c929c `core_v41_engram.c:66-93`:
    /// an image pads itself and every older n-gram position. Missing mask
    /// entries are text, like the engine's zero-filled history mask.
    pub(crate) fn rows_masked(&self, hist: &[i32], image_mask: &[u8], p: i64, ei: u32) -> Vec<i64> {
        let g = self.max_ngram as usize;
        let h = self.heads as usize;
        let mut prod = [0i64; 8];
        let mut blocked = false;
        for k in 0..g {
            let back = p - k as i64;
            let cid: i64 = if back < 0 || blocked {
                i64::from(self.pad)
            } else if image_mask.get(back as usize).is_some_and(|&v| v != 0) {
                blocked = true;
                i64::from(self.pad)
            } else {
                let tok = hist[back as usize];
                if tok >= 0 && (tok as u32) < self.n_vocab {
                    i64::from(self.token_map[tok as usize])
                } else {
                    i64::from(self.pad)
                }
            };
            let m = self.multipliers[ei as usize * g + k];
            // C `(int64_t)((uint64_t)cid * (uint64_t)mult)`: wrap-around
            // unsigned multiply, reinterpreted signed.
            prod[k] = (cid as u64).wrapping_mul(m as u64) as i64;
        }
        let mut rows = Vec::with_capacity((g - 1) * h);
        let mut rolling = prod[0];
        for i in 1..g {
            rolling ^= prod[i];
            for head in 0..h {
                let pr = self.primes[(ei as usize * (g - 1) + (i - 1)) * h + head];
                let mut r = rolling % pr;
                if r < 0 {
                    r += pr;
                }
                rows.push(r + self.offsets[ei as usize * (g - 1) * h + (i - 1) * h + head]);
            }
        }
        rows
    }
}

/// One row's raw bytes: `head_dim` fp8 weights and `head_dim/32` ue8m0
/// scales, from two separate planes in the shard.
pub struct EngramRow {
    pub weights: Vec<u8>,
    pub scale: Vec<u8>,
}

pub struct EngramShard {
    file: File,
    dio: bool,
    size: u64,
}

#[cfg(target_os = "linux")]
mod sys {
    use std::os::raw::c_int;

    /// `<arch>/include/uapi/asm/fcntl.h`: aarch64 defines its own O_DIRECT
    /// (0o200000, with 0o40000 being O_DIRECTORY there), while x86_64 and the
    /// asm-generic ports use 0o40000. The wrong value fails the open with
    /// ENOTDIR rather than silently doing nothing, which is how the Spark
    /// caught it: the fallback path ran and reported `direct=false`.
    #[cfg(target_arch = "aarch64")]
    pub const O_DIRECT: c_int = 0o200000;
    #[cfg(not(target_arch = "aarch64"))]
    pub const O_DIRECT: c_int = 0o40000;

    pub const POSIX_FADV_RANDOM: c_int = 1;

    extern "C" {
        pub fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
    }
}

impl EngramShard {
    /// C `v41_engram_open_shard` (`core_v41_engram.c:29-72`): O_DIRECT first,
    /// then plain read plus `FADV_RANDOM`, reporting which one was taken. The
    /// fallback is a real reproducibility risk, not a performance note.
    pub fn open(path: &Path) -> Result<Self, EngramError> {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let direct = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(sys::O_DIRECT)
                .open(path);
            if let Ok(file) = direct {
                let size = file
                    .metadata()
                    .map_err(|e| EngramError::Io(e.to_string()))?
                    .len();
                return Ok(Self {
                    file,
                    dio: true,
                    size,
                });
            }
        }
        let file =
            File::open(path).map_err(|e| EngramError::Open(format!("{}: {e}", path.display())))?;
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            unsafe {
                let _ = sys::posix_fadvise(file.as_raw_fd(), 0, 0, sys::POSIX_FADV_RANDOM);
            }
        }
        let size = file
            .metadata()
            .map_err(|e| EngramError::Io(e.to_string()))?
            .len();
        Ok(Self {
            file,
            dio: false,
            size,
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Bind checkpoint identity to the file this feed actually reads, even
    /// if its pathname is replaced after open.
    pub(crate) fn metadata(&self) -> Result<std::fs::Metadata, EngramError> {
        self.file
            .metadata()
            .map_err(|e| EngramError::Io(e.to_string()))
    }

    pub fn is_direct(&self) -> bool {
        self.dio
    }

    /// The shard's fd for the io_uring round (`v41_ering_req.fd`,
    /// core_v41.h:260); the ring owns no fd of its own.
    pub(crate) fn raw_fd(&self) -> std::os::raw::c_int {
        use std::os::unix::io::AsRawFd;
        self.file.as_raw_fd()
    }

    /// C `v41_ejob_create`'s span check (`core_v41_engram.c:236-242`).
    pub fn check_span(
        &self,
        weight_off: u64,
        scale_off: u64,
        rows: u64,
        head_dim: u64,
    ) -> Result<(), EngramError> {
        let nsc = head_dim / 32;
        if weight_off + rows * head_dim > self.size {
            return Err(EngramError::OutOfRange(0));
        }
        if scale_off + rows * nsc > self.size {
            return Err(EngramError::OutOfRange(1));
        }
        Ok(())
    }

    /// Read one row: `head_dim` weight bytes and `head_dim/32` scale bytes.
    pub fn row(
        &self,
        weight_off: u64,
        scale_off: u64,
        row: u64,
        head_dim: u32,
    ) -> Result<EngramRow, EngramError> {
        let hd = head_dim as u64;
        let nsc = hd / 32;
        let mut weights = vec![0u8; hd as usize];
        let mut scale = vec![0u8; nsc as usize];
        self.read_span(&mut weights, weight_off + row * hd)?;
        self.read_span(&mut scale, scale_off + row * nsc)?;
        Ok(EngramRow { weights, scale })
    }

    /// C `v41_edio_pread` (`core_v41_engram.c:24-27`): an O_DIRECT read must
    /// start and end on a block boundary, so read the aligned superset into a
    /// bounce buffer and copy out the requested span.
    pub fn read_span(&self, dst: &mut [u8], off: u64) -> Result<(), EngramError> {
        if !self.dio {
            return self
                .file
                .read_exact_at(dst, off)
                .map_err(|e| EngramError::Io(e.to_string()));
        }
        let a = off & !(EDIO_ALIGN as u64 - 1);
        let span = (off + dst.len() as u64 - a) as usize;
        let nb = span.div_ceil(EDIO_ALIGN) * EDIO_ALIGN;
        let mut bounce = Bounce::new(nb).ok_or_else(|| EngramError::Io("bounce alloc".into()))?;
        let buf = bounce.as_mut_slice();
        self.file
            .read_exact_at(buf, a)
            .map_err(|e| EngramError::Io(e.to_string()))?;
        dst.copy_from_slice(&buf[(off - a) as usize..(off - a) as usize + dst.len()]);
        Ok(())
    }
}

/// A 4096-aligned buffer for O_DIRECT reads (the engine's `posix_memalign`
/// bounce, `core_v41_engram.c:214-215`).
struct Bounce {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
}

impl Bounce {
    fn new(len: usize) -> Option<Self> {
        let layout = std::alloc::Layout::from_size_align(len, EDIO_ALIGN).ok()?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        Some(Self {
            ptr: std::ptr::NonNull::new(ptr)?,
            len,
        })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Bounce {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, EDIO_ALIGN).unwrap();
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

// ---- the interleaved sidecar (L3) -----------------------------------------
//
// The table above keeps the two planes apart (`core_v41_engram.c:236-242`):
// one read for `head_dim` weight bytes and one for `head_dim/32` scale bytes,
// 48 reads per token per layer.  The round is IOPS-bound, so the sidecar packs
// each row as one record (weights then scales) and halves the reads.  It is a
// derived copy of the same bytes, written by `examples/ds41_engram_pack.rs`;
// a layer's record stream is byte-for-byte the interleave of its two planes.

pub const SIDECAR_MAGIC: [u8; 8] = *b"DS41ENGM";
pub const SIDECAR_VERSION: u32 = 1;
/// magic + version + head_dim + layers + record bytes + reserved.
pub const SIDECAR_FIXED: usize = 32;
/// rows + record offset + source length + source mtime + source name hash.
pub const SIDECAR_LAYER: usize = 40;

/// One layer's entry: how many records it has, where they start, and the
/// source-plane stamp the packer saw (a stale sidecar is refused).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SidecarLayer {
    pub rows: u64,
    pub offset: u64,
    pub source: SidecarSource,
}

/// The identity of a source plane file the packer read.  The name hash rather
/// than the path lets `apply_engram_dir` swap directories, exactly as the
/// wire's table paths do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SidecarSource {
    pub len: u64,
    pub mtime_ns: u64,
    pub name: u64,
}

impl SidecarSource {
    pub fn of(path: &Path) -> Result<Self, EngramError> {
        let meta = std::fs::metadata(path)
            .map_err(|e| EngramError::Open(format!("{}: {e}", path.display())))?;
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as u64);
        let name = path.file_name().map_or_else(
            || path.to_string_lossy().into_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for b in name.bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        Ok(Self {
            len: meta.len(),
            mtime_ns,
            name: hash,
        })
    }
}

pub struct EngramSidecar {
    shard: EngramShard,
    head_dim: u32,
    rec_bytes: u32,
    layers: Vec<SidecarLayer>,
}

impl EngramSidecar {
    /// Opens and validates the header only; `matches` checks it against the
    /// wire before the feed reads a record.
    pub fn open(path: &Path) -> Result<Self, EngramError> {
        let shard = EngramShard::open(path)?;
        let mut fixed = [0u8; SIDECAR_FIXED];
        shard.read_span(&mut fixed, 0)?;
        if fixed[..8] != SIDECAR_MAGIC {
            return Err(EngramError::Open(format!(
                "{}: not an interleaved engram sidecar",
                path.display()
            )));
        }
        let version = u32::from_le_bytes(fixed[8..12].try_into().unwrap());
        if version != SIDECAR_VERSION {
            return Err(EngramError::Open(format!(
                "{}: sidecar version {version}, this build reads {SIDECAR_VERSION}",
                path.display()
            )));
        }
        let head_dim = u32::from_le_bytes(fixed[12..16].try_into().unwrap());
        let count = u32::from_le_bytes(fixed[16..20].try_into().unwrap());
        let rec_bytes = u32::from_le_bytes(fixed[20..24].try_into().unwrap());
        if head_dim == 0 || head_dim % 32 != 0 || rec_bytes != head_dim + head_dim / 32 {
            return Err(EngramError::Open(format!(
                "{}: head_dim {head_dim}, record {rec_bytes}",
                path.display()
            )));
        }
        let mut raw = vec![0u8; count as usize * SIDECAR_LAYER];
        shard.read_span(&mut raw, SIDECAR_FIXED as u64)?;
        let mut layers = Vec::with_capacity(count as usize);
        for b in raw.chunks_exact(SIDECAR_LAYER) {
            layers.push(SidecarLayer {
                rows: u64::from_le_bytes(b[0..8].try_into().unwrap()),
                offset: u64::from_le_bytes(b[8..16].try_into().unwrap()),
                source: SidecarSource {
                    len: u64::from_le_bytes(b[16..24].try_into().unwrap()),
                    mtime_ns: u64::from_le_bytes(b[24..32].try_into().unwrap()),
                    name: u64::from_le_bytes(b[32..40].try_into().unwrap()),
                },
            });
        }
        Ok(Self {
            shard,
            head_dim,
            rec_bytes,
            layers,
        })
    }

    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    pub fn rec_bytes(&self) -> u32 {
        self.rec_bytes
    }

    pub fn layers(&self) -> &[SidecarLayer] {
        &self.layers
    }

    pub fn raw_fd(&self) -> std::os::raw::c_int {
        self.shard.raw_fd()
    }

    pub fn is_direct(&self) -> bool {
        self.shard.is_direct()
    }

    pub fn size(&self) -> u64 {
        self.shard.size()
    }

    pub(crate) fn metadata(&self) -> Result<std::fs::Metadata, EngramError> {
        self.shard.metadata()
    }

    /// One row's record offset: the ring reads `rec_bytes` from here.
    pub fn record_off(&self, layer: usize, row: u64) -> Option<u64> {
        let l = self.layers.get(layer)?;
        (row < l.rows).then(|| l.offset + row * self.rec_bytes as u64)
    }

    /// Bulk read of `rows` consecutive records into `out` (`rows * rec_bytes`):
    /// the packer's verification pass and any bulk consumer.
    pub fn read_records(
        &self,
        layer: usize,
        row: u64,
        rows: u64,
        out: &mut [u8],
    ) -> Result<(), EngramError> {
        let l = self
            .layers
            .get(layer)
            .ok_or(EngramError::OutOfRange(layer))?;
        if row + rows > l.rows {
            return Err(EngramError::OutOfRange(layer));
        }
        if out.len() != rows as usize * self.rec_bytes as usize {
            return Err(EngramError::Io("record buffer size".into()));
        }
        self.shard
            .read_span(out, l.offset + row * self.rec_bytes as u64)
    }

    /// The row as the planes would return it: `head_dim` weights then
    /// `head_dim/32` scales, exactly the slot layout the feed already fills.
    pub fn row(&self, layer: usize, row: u64, head_dim: u32) -> Result<EngramRow, EngramError> {
        if head_dim != self.head_dim {
            return Err(EngramError::Open(format!(
                "sidecar head_dim {} != {head_dim}",
                self.head_dim
            )));
        }
        let off = self
            .record_off(layer, row)
            .ok_or(EngramError::OutOfRange(layer))?;
        let mut rec = vec![0u8; self.rec_bytes as usize];
        self.shard.read_span(&mut rec, off)?;
        let hd = self.head_dim as usize;
        Ok(EngramRow {
            weights: rec[..hd].to_vec(),
            scale: rec[hd..].to_vec(),
        })
    }

    /// Whether this sidecar was packed from the sources the wire points at:
    /// same head_dim, same row counts, same per-layer source stamps.
    pub fn matches(&self, head_dim: u32, rows: &[u64], sources: &[SidecarSource]) -> bool {
        self.head_dim == head_dim
            && rows.len() == self.layers.len()
            && sources.len() == self.layers.len()
            && self
                .layers
                .iter()
                .zip(rows)
                .zip(sources)
                .all(|((l, &r), &s)| l.rows == r && l.source == s)
    }
}

#[cfg(test)]
mod tests {
    use super::EngramHash;

    const HIST: [i32; 8] = [5, 6, 1, 3, 4, 0, 2, 7];
    const IMAGE_MASK: [u8; 8] = [0, 0, 1, 1, 0, 0, 0, 0];

    fn image_hash() -> EngramHash {
        EngramHash {
            token_map: (0..8).map(|i| i * 7 + 3).collect(),
            multipliers: (0..8).map(|i| i * 13 + 5).collect(),
            primes: (0..12).map(|i| 101 + i * 17).collect(),
            offsets: (0..12).map(|i| i * 1000).collect(),
            max_ngram: 4,
            heads: 2,
            pad: 2,
            n_vocab: 8,
        }
    }

    #[test]
    fn hash_stops_at_image() {
        let hash = image_hash();
        // Exact YoungAi 846c929c hash output, generated by the adjacent C
        // fixture; cover image queries and the three following positions.
        let fixture = include_str!("../tests/fixtures/engram/engram_mask_ref.txt");
        let mut checked = 0;
        for line in fixture.lines() {
            let mut values = line.split_whitespace().map(|v| v.parse::<i64>().unwrap());
            let ei = values.next().unwrap() as u32;
            let p = values.next().unwrap();
            let expected: Vec<i64> = values.collect();
            for mask in [&IMAGE_MASK[..], &IMAGE_MASK[..4]] {
                assert_eq!(
                    hash.rows_masked(&HIST, mask, p, ei),
                    expected,
                    "ei={ei} p={p} mask_len={}",
                    mask.len()
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 2 * HIST.len());
    }
}

#[cfg(test)]
mod sidecar_tests {
    use super::*;

    fn build(path: &Path, head_dim: u32, rows: &[u64]) {
        let rec = (head_dim + head_dim / 32) as usize;
        let start = 4096u64;
        let mut bytes = vec![0u8; SIDECAR_FIXED + rows.len() * SIDECAR_LAYER];
        bytes[..8].copy_from_slice(&SIDECAR_MAGIC);
        bytes[8..12].copy_from_slice(&SIDECAR_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&head_dim.to_le_bytes());
        bytes[16..20].copy_from_slice(&(rows.len() as u32).to_le_bytes());
        bytes[20..24].copy_from_slice(&(rec as u32).to_le_bytes());
        let mut at = start;
        for (i, &n) in rows.iter().enumerate() {
            let e = SIDECAR_FIXED + i * SIDECAR_LAYER;
            bytes[e..e + 8].copy_from_slice(&n.to_le_bytes());
            bytes[e + 8..e + 16].copy_from_slice(&at.to_le_bytes());
            bytes[e + 16..e + 24].copy_from_slice(&1u64.to_le_bytes());
            bytes[e + 24..e + 32].copy_from_slice(&2u64.to_le_bytes());
            bytes[e + 32..e + 40].copy_from_slice(&3u64.to_le_bytes());
            at += n * rec as u64;
        }
        bytes.resize(start as usize, 0);
        for row in 0..rows.iter().sum::<u64>() {
            let mut rec_bytes = vec![0u8; rec];
            rec_bytes[..head_dim as usize].fill(row as u8);
            rec_bytes[head_dim as usize..].fill(0x80 | row as u8);
            bytes.extend_from_slice(&rec_bytes);
        }
        // The O_DIRECT bounce reads a whole aligned block past the last record.
        let padded = bytes.len().div_ceil(EDIO_ALIGN) * EDIO_ALIGN;
        bytes.resize(padded, 0);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn sidecar_reads_one_record_per_row() {
        let dir = std::env::temp_dir().join(format!("ds41-sidecar-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sidecar.bin");
        build(&path, 32, &[3, 2]);
        let sc = EngramSidecar::open(&path).unwrap();
        assert_eq!(sc.head_dim(), 32);
        assert_eq!(sc.rec_bytes(), 33);
        assert_eq!(sc.layers().len(), 2);

        let row = sc.row(0, 2, 32).unwrap();
        assert!(row.weights.iter().all(|&b| b == 2));
        assert_eq!(row.scale, vec![0x82]);

        // Layer 1's first record follows layer 0's three.
        assert_eq!(sc.record_off(1, 0), Some(4096 + 3 * 33));
        let row = sc.row(1, 1, 32).unwrap();
        assert!(row.weights.iter().all(|&b| b == 4));
        assert_eq!(row.scale, vec![0x84]);

        // Out-of-range rows and a head_dim that does not match are refused.
        assert!(sc.record_off(1, 2).is_none());
        assert!(sc.row(0, 0, 64).is_err());

        let mut buf = vec![0u8; 2 * 33];
        sc.read_records(0, 1, 2, &mut buf).unwrap();
        assert!(buf[..32].iter().all(|&b| b == 1));
        assert!(buf[33..65].iter().all(|&b| b == 2));
        assert!(sc.read_records(0, 2, 2, &mut buf).is_err());

        let sources = [
            SidecarSource {
                len: 1,
                mtime_ns: 2,
                name: 3,
            },
            SidecarSource {
                len: 1,
                mtime_ns: 2,
                name: 3,
            },
        ];
        assert!(sc.matches(32, &[3, 2], &sources));
        assert!(!sc.matches(32, &[3, 3], &sources));
        assert!(!sc.matches(64, &[3, 2], &sources));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn sidecar_refuses_a_foreign_header() {
        let dir = std::env::temp_dir().join(format!("ds41-sidecar-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.bin");
        std::fs::write(&path, vec![0u8; 8192]).unwrap();
        let err = match EngramSidecar::open(&path) {
            Ok(_) => panic!("a zero header must be refused"),
            Err(e) => e.token(),
        };
        assert!(err.contains("not an interleaved engram sidecar"), "{err}");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
