//! MiniMax-M3.1-Flash VQ (`minimax_m3_vq`) is its own artifact contract.
//! Byte layout: Baekpica/MiniMax-M3.1-Flash-VQ-GGUF FORMAT.md at
//! 74e5a9fd12fb57a5c1babb559536cf10733939ea. Routed experts are 13-bit GVQ1
//! matrices (Beam's container: +0 gain on zero rows) with GLM's GVR8
//! residual stage (R13+8) on twenty layers. Besides the trunk the release
//! has an `embedding` part (the BF16 token table the host reads in rows), a
//! `vision` sidecar and the DSpark `drafter`. Validate borrowed mmap spans;
//! never expand experts or copy a part.

mod layout;
pub mod media;
mod model;
mod release;

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::gguf::GgufFile;
use crate::tensors::{TensorError, TensorInfo};

pub use layout::EMBEDDING;
pub use release::Feature;

const ARCH: &[u8] = b"minimax_m3_vq";
const NAMESPACE: &[u8] = b"minimax_m3_vq.";
const VARIANT: &[u8] = b"minimax-m3-vq8";
const SOURCE_MODEL: &[u8] = b"minimax_m3_vl";
/// Parts keep the recipe id written before the residual stage existed;
/// release.json names the final recipe (handoff HANDOFF.md, FORMAT.md).
const PART_RECIPE: &[u8] = b"minimax-m31f-vq8-r13x57-q4k-nvfp4idx-ssdkv-embssd-v1";
const RECIPE: &[u8] = b"minimax-m31f-vq8-r13x57-r8x20-q4k-nvfp4idx-ssdkv-embssd-v1";
const ALIGN: u64 = 128;
const HEADER: u64 = 128;
const VECTOR: u8 = 8;
const WORDS: u32 = 8192;
/// Container 1 holds R13 layers; 2 adds the GVR8 residual stage (R13+8).
const FORMAT_VERSIONS: std::ops::RangeInclusive<u32> = 1..=2;
const RESIDUAL_VERSION: u32 = 2;
const E4M3: u32 = 1;
const CODEBOOK: u32 = 0x4756_4342;
const VQ: u32 = 0x4756_5138;
const RESIDUAL: u32 = 0x4756_5238;
const RESIDUAL_BITS: u8 = 8;
const RESIDUAL_WORDS: u32 = 256;
const F32: u32 = 0;
const F16: u32 = 1;
const BF16: u32 = 30;
const Q8_0: u32 = 8;
const Q8_BLOCK: u64 = 32;
const Q8_BYTES: u64 = 34;
const Q4_K: u32 = 12;
const Q4_BLOCK: u64 = 256;
const Q4_BYTES: u64 = 144;

type Result<T> = std::result::Result<T, TensorError>;

thread_local! {
    /// Set while the release reader parses a part: its O_DIRECT hashing
    /// pass checks the quant scales, so the map never streams.
    static STREAMED: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` with the whole-tensor content scans left to the caller's stream.
fn streamed<T>(f: impl FnOnce() -> T) -> T {
    STREAMED.with(|s| s.set(true));
    let out = f();
    STREAMED.with(|s| s.set(false));
    out
}

/// Both f16 scales (d, dmin) at the start of a Q4_K block are finite.
fn q4_scales_finite(scales: &[u8]) -> bool {
    scales[..4]
        .chunks_exact(2)
        .all(|s| u16::from_le_bytes([s[0], s[1]]) & 0x7c00 != 0x7c00)
}

/// The f16 scale at the start of a Q8_0 block is finite.
fn q8_scale_finite(scale: &[u8]) -> bool {
    u16::from_le_bytes([scale[0], scale[1]]) & 0x7c00 != 0x7c00
}

/// The release's vision part (image and video input), when it ships one.
pub(crate) fn vision_path(root: &std::path::Path) -> Option<std::path::PathBuf> {
    release::vision_path(root).ok().flatten()
}

/// The DSpark drafter rides along when speculative decoding is requested
/// (`--mtp-draft` above 1), as Ling's embedded MTP part does.
pub(super) fn feature(draft: i32) -> Feature {
    let disabled = std::env::var("DS4_MTP_SPEC_DISABLE").ok().as_deref() == Some("1");
    if draft > 1 && !disabled {
        Feature::Draft
    } else {
        Feature::Trunk
    }
}

/// The device-copied parts (trunk, drafter on request), then the embedding
/// part, whose rows the host reads per token, last.
pub(super) fn open_dir(
    path: &std::path::Path,
    feature: Feature,
) -> Result<crate::tensors::TensorInventory> {
    release::open_dir(path, feature)
}

/// A directory whose `release.json` names this format.
pub(super) fn is_release(path: &std::path::Path) -> bool {
    release::is_release(path)
}

pub(super) fn identify(
    path: &std::path::Path,
) -> std::result::Result<crate::Identified, crate::IdentifyError> {
    model::identify(path)
}

pub(super) fn layouts() -> Vec<crate::LayoutSpec> {
    layout::layouts()
}

pub(super) fn asset(root: &std::path::Path, name: &str) -> Result<Vec<u8>> {
    release::asset(root, name)
}

/// Bytes of the device-resident parts (the embedding and vision parts stay
/// on SSD; the drafter only with `Feature::Draft`).
pub(super) fn resident_bytes(root: &std::path::Path, feature: Feature) -> Result<u64> {
    release::resident_bytes(root, feature)
}

pub(super) fn metadata(root: &std::path::Path) -> Result<GgufFile> {
    Ok(GgufFile::open(&release::first_part(root)?)?)
}

pub(super) fn validate_inventory(
    inventory: &crate::TensorInventory,
) -> std::result::Result<(), crate::ValidateError> {
    let plan = crate::BindPlan::resolve(crate::shape::SHAPE_MINIMAX_VQ, inventory);
    let found = plan.slots.iter().filter(|s| s.tensor.is_some()).count();
    if found != inventory.tensors.len() || !plan.missing_required().is_empty() {
        return Err(crate::ValidateError::Token(
            "MiniMax VQ tensor set mismatch",
        ));
    }
    crate::validate_layouts(&plan)
        .map_err(|e| crate::ValidateError::TokenKey("MiniMax VQ tensor layout", e.to_string()))
}

fn fail(reason: &'static str) -> TensorError {
    TensorError::MinimaxVq(reason)
}

fn span(data: &[u8], start: u64, len: u64) -> Result<&[u8]> {
    let end = start.checked_add(len).ok_or(TensorError::Overflow)?;
    let start = usize::try_from(start).map_err(|_| TensorError::Overflow)?;
    let end = usize::try_from(end).map_err(|_| TensorError::Overflow)?;
    data.get(start..end).ok_or(TensorError::OutsideFile)
}

fn align(value: u64) -> Result<u64> {
    value
        .checked_add(ALIGN - 1)
        .map(|n| n / ALIGN * ALIGN)
        .ok_or(TensorError::Overflow)
}

fn zeros(data: &[u8], start: u64, end: u64) -> Result<()> {
    if end < start || span(data, start, end - start)?.iter().any(|&x| x != 0) {
        return Err(fail("nonzero padding"));
    }
    Ok(())
}

fn u32_at(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off + 4].try_into().unwrap())
}

fn u64_at(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
}

/// GVQ1 matrix header (FORMAT.md "GVQ1 matrix"): K8192 books only, so the
/// 12-bit index stream always has its separate 13th-bit plane.
struct Header {
    book: u32,
    size: u64,
}

impl Header {
    fn parse(data: &[u8], tensor: &TensorInfo) -> Result<Self> {
        let h = span(data, tensor.abs_offset, HEADER)?;
        let rows = u64::from(u32_at(h, 8));
        let cols = u64::from(u32_at(h, 12));
        if &h[..4] != b"GVQ1"
            || h[4..6] != [1, 0]
            || h[6] != 13
            || h[7] != VECTOR
            || u32_at(h, 16) != WORDS
            || u32_at(h, 20) != E4M3
            || u32_at(h, 28) != 0
            || rows == 0
            || cols == 0
            || cols % 64 != 0
            || tensor.ndim != 2
            || tensor.dim[..2] != [cols, rows]
        {
            return Err(fail("invalid GVQ1 header/shape"));
        }
        let elems = rows.checked_mul(cols).ok_or(TensorError::Overflow)?;
        let main = align(HEADER + 2 * rows)?;
        let main_end = main
            .checked_add(elems / 16 * 3)
            .ok_or(TensorError::Overflow)?;
        // The high plane has its own row stride (cols/64), not main's 3*cols/16.
        let plane = align(main_end)?;
        let end = plane.checked_add(elems / 64).ok_or(TensorError::Overflow)?;
        let size = align(end)?;
        if [u64_at(h, 32), u64_at(h, 40), u64_at(h, 48), u64_at(h, 56)]
            != [HEADER, main, plane, size]
        {
            return Err(fail("noncanonical GVQ1 offsets"));
        }
        let payload = span(data, tensor.abs_offset, size)?;
        for gain in span(payload, HEADER, 2 * rows)?.chunks_exact(2) {
            let bits = u16::from_le_bytes(gain.try_into().unwrap());
            // +0 is the zero-row encoding; -0 and negatives are invalid.
            if bits & 0x8000 != 0 || bits & 0x7c00 == 0x7c00 {
                return Err(fail("negative/nonfinite row gain"));
            }
        }
        zeros(payload, HEADER + 2 * rows, main)?;
        zeros(payload, main_end, plane)?;
        zeros(payload, end, size)?;
        Ok(Self {
            book: u32_at(h, 24),
            size,
        })
    }
}

/// GVR8: one index byte per 8-weight vector into a K=256 E4M3 book; the
/// parent GVQ1 matrix supplies the row gain (GLM FORMAT.md "Residual
/// stage"). Canonical 128-byte layout.
struct ResidualHeader {
    book: u32,
    size: u64,
}

impl ResidualHeader {
    fn parse(data: &[u8], tensor: &TensorInfo) -> Result<Self> {
        let h = span(data, tensor.abs_offset, HEADER)?;
        let rows = u64::from(u32_at(h, 8));
        let cols = u64::from(u32_at(h, 12));
        if &h[..4] != b"GVR1"
            || h[4..6] != [1, 0]
            || h[6] != RESIDUAL_BITS
            || h[7] != VECTOR
            || u32_at(h, 16) != RESIDUAL_WORDS
            || u32_at(h, 20) != E4M3
            || u32_at(h, 28) != 0
            || rows == 0
            || cols == 0
            || cols % 64 != 0
            || tensor.ndim != 2
            || tensor.dim[..2] != [cols, rows]
        {
            return Err(fail("invalid GVR1 header/shape"));
        }
        let end = rows
            .checked_mul(cols / u64::from(VECTOR))
            .and_then(|n| n.checked_add(HEADER))
            .ok_or(TensorError::Overflow)?;
        let size = align(end)?;
        if [u64_at(h, 32), u64_at(h, 40), u64_at(h, 48), u64_at(h, 56)] != [HEADER, 0, 0, size] {
            return Err(fail("noncanonical GVR1 offsets"));
        }
        zeros(span(data, tensor.abs_offset, size)?, end, size)?;
        Ok(Self {
            book: u32_at(h, 24),
            size,
        })
    }
}

/// A payload's codebook reference: tensor index, book ID and words.
struct BookLink {
    tensor: usize,
    book: u32,
    words: u32,
}

fn object(g: &GgufFile, key: &str) -> Result<Map<String, Value>> {
    let Some(bytes) = g.get_string(key) else {
        if g.kv_entries()
            .iter()
            .any(|e| g.key_bytes(e) == key.as_bytes())
        {
            return Err(fail("link metadata must be JSON strings"));
        }
        return Ok(Map::new());
    };
    match serde_json::from_slice(bytes) {
        Ok(Value::Object(map)) => Ok(map),
        _ => Err(fail("invalid link object")),
    }
}

fn raw<'a>(data: &'a [u8], tensor: &TensorInfo) -> Result<&'a [u8]> {
    span(data, tensor.abs_offset, tensor.bytes)
}

fn validate_books(g: &GgufFile, tensors: &[TensorInfo], links: &[BookLink]) -> Result<()> {
    let books = object(g, "minimax_m3_vq.codebooks")?;
    let by_name: HashMap<_, _> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut used = HashSet::new();
    let mut hashes = HashMap::new();
    for link in links {
        let id = link.book.to_string();
        let name = books
            .get(&id)
            .and_then(Value::as_str)
            .ok_or(fail("missing codebook link"))?;
        let book = by_name.get(name).ok_or(fail("missing codebook tensor"))?;
        if book.typ != CODEBOOK
            || book.ndim != 2
            || book.dim[..2] != [u64::from(VECTOR), u64::from(link.words)]
        {
            return Err(fail("codebook type/shape"));
        }
        let hash = match hashes.get(name) {
            Some(hash) => *hash,
            None => {
                let bytes = raw(g.as_bytes(), book)?;
                // E4M3FN signed NaNs 0x7f/0xff are rejected (FORMAT.md).
                if bytes.iter().any(|x| x & 0x7f == 0x7f) {
                    return Err(fail("nonfinite codebook"));
                }
                let hash: [u8; 32] = Sha256::digest(bytes).into();
                hashes.insert(name, hash);
                hash
            }
        };
        // The payload digest (header 96..128) is the converter's own check:
        // release.json's part SHA256 already covers every payload byte.
        if raw(g.as_bytes(), &tensors[link.tensor])?[64..96] != hash {
            return Err(fail("codebook checksum"));
        }
        used.insert(id);
    }
    if used.len() != books.len()
        || tensors
            .iter()
            .any(|t| t.typ == CODEBOOK && !hashes.contains_key(t.name.as_str()))
    {
        return Err(fail("unused codebook/link"));
    }
    Ok(())
}

/// `minimax_m3_vq.residuals` maps each GVQ1 matrix of an R13+8 part to its
/// GVR8 tensor. A part's residual stage is all-or-none: every matrix of the
/// layer has exactly one residual and every residual one parent.
fn validate_residuals(g: &GgufFile, tensors: &[TensorInfo], version: u32) -> Result<()> {
    let links = object(g, "minimax_m3_vq.residuals")?;
    let index: HashMap<_, _> = tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();
    let residuals = tensors.iter().filter(|t| t.typ == RESIDUAL).count();
    if residuals == 0 && links.is_empty() {
        if version == RESIDUAL_VERSION {
            return Err(fail("container version 2 without a residual stage"));
        }
        return Ok(());
    }
    if version != RESIDUAL_VERSION {
        return Err(fail("residual stage needs container version 2"));
    }
    let mut linked = HashSet::new();
    for (parent, value) in &links {
        let name = value.as_str().ok_or(fail("invalid residual link"))?;
        let (Some(&p), Some(&r)) = (index.get(parent.as_str()), index.get(name)) else {
            return Err(fail("missing residual parent/tensor"));
        };
        let (pt, rt) = (&tensors[p], &tensors[r]);
        if pt.typ != VQ || rt.typ != RESIDUAL || pt.dim[..2] != rt.dim[..2] || !linked.insert(r) {
            return Err(fail("residual link type/shape/duplicate"));
        }
    }
    let parents = tensors.iter().filter(|t| t.typ == VQ).count();
    if linked.len() != residuals || residuals != parents {
        return Err(fail("partial or unlinked residual stage"));
    }
    Ok(())
}

fn part_metadata(g: &GgufFile) -> Result<u32> {
    let version = g.get_u32("minimax_m3_vq.format_version").unwrap_or(0);
    if g.get_string("general.architecture") != Some(ARCH)
        || g.alignment != ALIGN
        || !FORMAT_VERSIONS.contains(&version)
        || g.get_string("minimax_m3_vq.variant") != Some(VARIANT)
        || g.get_string("minimax_m3_vq.source_model_type") != Some(SOURCE_MODEL)
        || g.get_string("minimax_m3_vq.recipe_id") != Some(PART_RECIPE)
        || !matches!(
            g.get_string("minimax_m3_vq.part_role"),
            Some(b"trunk") | Some(b"embedding") | Some(b"vision") | Some(b"drafter")
        )
        || g.split_count() > 1
    {
        return Err(fail("unsupported format/version/recipe/alignment"));
    }
    let mut keys = HashSet::new();
    for entry in g.kv_entries() {
        let key = g.key_bytes(entry);
        if !keys.insert(key) || !(key.starts_with(b"general.") || key.starts_with(NAMESPACE)) {
            return Err(fail("duplicate/foreign metadata"));
        }
    }
    Ok(version)
}

/// Whether this part opts into the MiniMax validator (checked before
/// Beam's, Ling's and GLM's, which would claim its GVQ1 tensors).
pub(super) fn requested(g: &GgufFile) -> bool {
    g.get_string("general.architecture") == Some(ARCH)
        || g.kv_entries()
            .iter()
            .any(|e| g.key_bytes(e).starts_with(NAMESPACE))
}

/// Payload bytes of a block-quantized matrix: GGUF dim 0 is the input width.
fn quant_bytes(t: &TensorInfo, block: u64, bytes: u64) -> Result<u64> {
    if t.ndim != 2 || t.dim[0] % block != 0 {
        return Err(fail("unsupported quant shape"));
    }
    t.elements
        .checked_div(block)
        .and_then(|n| n.checked_mul(bytes))
        .ok_or(TensorError::Overflow)
}

/// One independent part: metadata, sizes, geometry, links and padding.
pub(super) fn validate_part(
    g: &GgufFile,
    tensors: &mut [TensorInfo],
    dir_end: usize,
    data_pos: u64,
) -> Result<()> {
    let version = part_metadata(g)?;
    let mut names = HashSet::new();
    let mut links = Vec::new();
    for (i, t) in tensors.iter_mut().enumerate() {
        if !names.insert(t.name.clone())
            || t.ndim > 4
            || t.dim[..t.ndim as usize].contains(&0)
            || t.rel_offset % ALIGN != 0
        {
            return Err(fail("invalid tensor name/shape/alignment"));
        }
        t.bytes = match t.typ {
            F32 => t.elements.checked_mul(4).ok_or(TensorError::Overflow)?,
            F16 | BF16 => t.elements.checked_mul(2).ok_or(TensorError::Overflow)?,
            CODEBOOK => t.elements,
            Q4_K => quant_bytes(t, Q4_BLOCK, Q4_BYTES)?,
            Q8_0 => quant_bytes(t, Q8_BLOCK, Q8_BYTES)?,
            VQ => {
                let header = Header::parse(g.as_bytes(), t)?;
                links.push(BookLink {
                    tensor: i,
                    book: header.book,
                    words: WORDS,
                });
                header.size
            }
            RESIDUAL => {
                let header = ResidualHeader::parse(g.as_bytes(), t)?;
                links.push(BookLink {
                    tensor: i,
                    book: header.book,
                    words: RESIDUAL_WORDS,
                });
                header.size
            }
            _ => return Err(fail("unsupported tensor type")),
        };
        let bytes = raw(g.as_bytes(), t)?;
        if STREAMED.with(Cell::get) {
            continue;
        }
        if t.typ == Q4_K && !bytes.chunks_exact(Q4_BYTES as usize).all(q4_scales_finite) {
            return Err(fail("nonfinite Q4_K scale"));
        }
        if t.typ == Q8_0 && !bytes.chunks_exact(Q8_BYTES as usize).all(q8_scale_finite) {
            return Err(fail("nonfinite Q8_0 scale"));
        }
    }
    zeros(g.as_bytes(), dir_end as u64, data_pos)?;
    let mut order: Vec<_> = tensors.iter().collect();
    order.sort_unstable_by_key(|t| t.abs_offset);
    let mut end = data_pos;
    for tensor in order {
        if tensor.abs_offset != align(end)? {
            return Err(fail("gap/overlap"));
        }
        zeros(g.as_bytes(), end, tensor.abs_offset)?;
        end = tensor
            .abs_offset
            .checked_add(tensor.bytes)
            .ok_or(TensorError::Overflow)?;
    }
    if end != g.as_bytes().len() as u64 {
        return Err(fail("trailing/missing bytes"));
    }
    validate_books(g, tensors, &links)?;
    validate_residuals(g, tensors, version)
}
