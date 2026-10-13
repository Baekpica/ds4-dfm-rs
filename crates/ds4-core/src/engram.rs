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

use crate::gguf::GgufFile;
use crate::tensors::TensorInventory;
use crate::v41::V41Wire;

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

    /// C `v41_engram_hash` (`core_v41_engram.c:94-116`). `hist` is the token
    /// history by absolute position; `p` is the absolute position to hash;
    /// `ei` is the engram index (not the layer number).
    pub fn rows(&self, hist: &[i32], p: i64, ei: u32) -> Vec<i64> {
        let g = self.max_ngram as usize;
        let h = self.heads as usize;
        let mut prod = [0i64; 8];
        for k in 0..g {
            let back = p - k as i64;
            let cid: i64 = if back < 0 {
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
        self.read_exact(&mut weights, weight_off + row * hd)?;
        self.read_exact(&mut scale, scale_off + row * nsc)?;
        Ok(EngramRow { weights, scale })
    }

    /// C `v41_edio_pread` (`core_v41_engram.c:24-27`): an O_DIRECT read must
    /// start and end on a block boundary, so read the aligned superset into a
    /// bounce buffer and copy out the requested span.
    fn read_exact(&self, dst: &mut [u8], off: u64) -> Result<(), EngramError> {
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
