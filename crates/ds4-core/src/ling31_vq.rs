//! Ling-3.1-flash VQ (`ling31_vq`, GVQ1 v1) is its own artifact contract.
//! Byte layout: Baekpica/Ling-3.1-flash-VQ-GGUF FORMAT.md at
//! c229b706131dcf200d4569cbfab1783aa9977301. GVQ1 matrices keep GLM's v1
//! geometry, but Ling stores exact zero rows as +0 gain, K4096 layers have
//! no high plane, and FP8 scales cover partial 128-blocks by ceil division.
//! Validate borrowed mmap spans; never expand experts or copy a part.

mod layout;
mod model;
mod release;

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::gguf::GgufFile;
use crate::tensors::{TensorError, TensorInfo};

pub use release::Feature;

const ARCH: &[u8] = b"ling31_vq";
const VARIANT: &[u8] = b"ling31-vq8";
const SOURCE_MODEL: &[u8] = b"bailing_hybrid";
const RECIPE: &[u8] = b"ling31-vq8-mix8r13-60r12-mtp13-q4h-v1";
const ALIGN: u64 = 128;
const HEADER: u64 = 128;
const VECTOR: u8 = 8;
const FORMAT_VERSION: u32 = 1;
const FP8: u32 = 0x4751_4638;
const CODEBOOK: u32 = 0x4756_4342;
const VQ: u32 = 0x4756_5138;
const F32: u32 = 0;
const BF16: u32 = 30;
const Q4_K: u32 = 12;
const Q4_BLOCK: u64 = 256;
const Q4_BYTES: u64 = 144;
const SCALE_BLOCK: u64 = 128;

type Result<T> = std::result::Result<T, TensorError>;

thread_local! {
    /// Set while the release reader parses a part: its O_DIRECT hashing
    /// pass checks Q4_K scales and FP8 weights, so the map never streams.
    static STREAMED: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` with the whole-tensor content scans left to the caller's stream.
pub(super) fn streamed<T>(f: impl FnOnce() -> T) -> T {
    STREAMED.with(|s| s.set(true));
    let out = f();
    STREAMED.with(|s| s.set(false));
    out
}

/// E4M3FN NaN encodings are rejected in the source weights.
pub(super) fn fp8_finite(bytes: &[u8]) -> bool {
    !bytes.iter().any(|x| x & 0x7f == 0x7f)
}

/// Both f16 scales (d, dmin) at the start of a Q4_K block are finite.
pub(super) fn q4_scales_finite(scales: &[u8]) -> bool {
    scales[..4]
        .chunks_exact(2)
        .all(|s| u16::from_le_bytes([s[0], s[1]]) & 0x7c00 != 0x7c00)
}

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

/// MTP drafting needs its part mapped: `--mtp-draft N` (N > 1) or
/// DS4_LING31_MTP=1, unless DS4_MTP_SPEC_DISABLE=1 (native ds4_engine_has_mtp).
pub(super) fn feature(draft: i32) -> Feature {
    let env = |name| std::env::var(name).ok();
    let requested = draft > 1 || env("DS4_LING31_MTP").as_deref() == Some("1");
    if requested && env("DS4_MTP_SPEC_DISABLE").as_deref() != Some("1") {
        Feature::Mtp
    } else {
        Feature::Trunk
    }
}

pub(super) fn part_bytes(root: &std::path::Path, feature: Feature) -> Result<u64> {
    release::part_bytes(root, feature)
}

pub(super) fn metadata(root: &std::path::Path) -> Result<GgufFile> {
    Ok(GgufFile::open(&release::first_part(root)?)?)
}

pub(super) fn validate_inventory(
    inventory: &crate::TensorInventory,
) -> std::result::Result<(), crate::ValidateError> {
    let plan = crate::BindPlan::resolve(crate::shape::SHAPE_LING31_VQ, inventory);
    let found = plan.slots.iter().filter(|s| s.tensor.is_some()).count();
    // The optional MTP block arrives whole (both mtp parts) or not at all.
    let mtp = plan
        .slots
        .iter()
        .filter(|s| !s.need.required())
        .map(|s| s.tensor.is_some())
        .collect::<HashSet<_>>();
    if found != inventory.tensors.len() || mtp.len() != 1 {
        return Err(crate::ValidateError::Token(
            "Ling-3.1 VQ tensor set mismatch",
        ));
    }
    crate::validate_layouts(&plan)
        .map_err(|e| crate::ValidateError::TokenKey("Ling-3.1 VQ tensor layout", e.to_string()))
}

fn fail(reason: &'static str) -> TensorError {
    TensorError::Ling31Vq(reason)
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

fn ceil_blocks(n: u64) -> u64 {
    n.div_ceil(SCALE_BLOCK)
}

/// GVQ1 matrix header (FORMAT.md "Matrix byte layout"). K4096 rows carry
/// 12-bit indices only; K8192 adds the separate 13th-bit plane.
struct Header {
    k: u32,
    book: u32,
    size: u64,
}

impl Header {
    fn parse(data: &[u8], tensor: &TensorInfo) -> Result<Self> {
        let h = span(data, tensor.abs_offset, HEADER)?;
        let rows = u64::from(u32_at(h, 8));
        let cols = u64::from(u32_at(h, 12));
        let k = u32_at(h, 16);
        if &h[..4] != b"GVQ1"
            || h[4..6] != [1, 0]
            || h[7] != VECTOR
            || !matches!(k, 4096 | 8192)
            || h[6] != if k == 4096 { 12 } else { 13 }
            // The release stores E4M3FN books only; F16 books are a converter control.
            || u32_at(h, 20) != 1
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
        let plane = if k == 8192 { align(main_end)? } else { 0 };
        let end = if plane != 0 {
            plane.checked_add(elems / 64).ok_or(TensorError::Overflow)?
        } else {
            main_end
        };
        let size = align(end)?;
        if [u64_at(h, 32), u64_at(h, 40), u64_at(h, 48), u64_at(h, 56)]
            != [HEADER, main, plane, size]
        {
            return Err(fail("noncanonical GVQ1 offsets"));
        }
        let payload = span(data, tensor.abs_offset, size)?;
        for gain in span(payload, HEADER, 2 * rows)?.chunks_exact(2) {
            let bits = u16::from_le_bytes(gain.try_into().unwrap());
            // +0 is the Ling zero-row encoding; -0 and negatives are invalid.
            if bits & 0x8000 != 0 || bits & 0x7c00 == 0x7c00 {
                return Err(fail("negative/nonfinite row gain"));
            }
        }
        zeros(payload, HEADER + 2 * rows, main)?;
        if plane != 0 {
            zeros(payload, main_end, plane)?;
        }
        zeros(payload, end, size)?;
        Ok(Self {
            k,
            book: u32_at(h, 24),
            size,
        })
    }
}

fn object(g: &GgufFile, key: &str) -> Result<Map<String, Value>> {
    let Some(bytes) = g.get_string(key) else {
        return Err(fail("missing link metadata"));
    };
    match serde_json::from_slice(bytes) {
        Ok(Value::Object(map)) => Ok(map),
        _ => Err(fail("invalid link object")),
    }
}

fn raw<'a>(data: &'a [u8], tensor: &TensorInfo) -> Result<&'a [u8]> {
    span(data, tensor.abs_offset, tensor.bytes)
}

fn validate_books(g: &GgufFile, tensors: &[TensorInfo], headers: &[(usize, Header)]) -> Result<()> {
    let links = object(g, "ling_vq.codebooks")?;
    let by_name: HashMap<_, _> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut used = HashSet::new();
    let mut hashes = HashMap::new();
    for (i, header) in headers {
        let id = header.book.to_string();
        let name = links
            .get(&id)
            .and_then(Value::as_str)
            .ok_or(fail("missing codebook link"))?;
        let book = by_name.get(name).ok_or(fail("missing codebook tensor"))?;
        if book.typ != CODEBOOK
            || book.ndim != 2
            || book.dim[..2] != [u64::from(VECTOR), u64::from(header.k)]
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
        // release.json's file SHA256 already covers every payload byte.
        if raw(g.as_bytes(), &tensors[*i])?[64..96] != hash {
            return Err(fail("codebook checksum"));
        }
        used.insert(id);
    }
    if used.len() != links.len()
        || tensors
            .iter()
            .any(|t| t.typ == CODEBOOK && !hashes.contains_key(t.name.as_str()))
    {
        return Err(fail("unused codebook/link"));
    }
    Ok(())
}

fn validate_native(g: &GgufFile, tensors: &[TensorInfo]) -> Result<()> {
    let by_name: HashMap<_, _> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let scales = object(g, "ling_vq.fp8_scales")?;
    let mut seen = 0;
    for t in tensors.iter().filter(|t| t.typ == FP8) {
        let name = scales
            .get(&t.name)
            .and_then(Value::as_str)
            .ok_or(fail("missing FP8 scale link"))?;
        let scale = by_name.get(name).ok_or(fail("missing FP8 scale tensor"))?;
        // Scale grid [ceil(R/128), ceil(C/128)]: [576,4096] has [5,32].
        if t.ndim != 2
            || scale.typ != F32
            || scale.ndim != 2
            || scale.dim[..2] != [ceil_blocks(t.dim[0]), ceil_blocks(t.dim[1])]
        {
            return Err(fail("FP8 scale type/shape"));
        }
        // weight_scale_inv is an original multiplicative F32 scale.
        for bytes in raw(g.as_bytes(), scale)?.chunks_exact(4) {
            let value = f32::from_le_bytes(bytes.try_into().unwrap());
            if !value.is_finite() || value <= 0.0 {
                return Err(fail("nonpositive/nonfinite FP8 scale"));
            }
        }
        if !STREAMED.with(Cell::get) && !fp8_finite(raw(g.as_bytes(), t)?) {
            return Err(fail("nonfinite FP8 weight"));
        }
        seen += 1;
    }
    if seen != scales.len() {
        return Err(fail("surplus FP8 scale links"));
    }
    Ok(())
}

fn part_metadata(g: &GgufFile) -> Result<()> {
    if g.get_string("general.architecture") != Some(ARCH)
        || g.alignment != ALIGN
        || g.get_u32("ling_vq.format_version") != Some(FORMAT_VERSION)
        || g.get_string("ling_vq.variant") != Some(VARIANT)
        || g.get_string("ling_vq.source_model_type") != Some(SOURCE_MODEL)
        || g.get_string("ling_vq.recipe_id") != Some(RECIPE)
        || !matches!(
            g.get_string("ling_vq.part_role"),
            Some(b"trunk") | Some(b"mtp")
        )
        || g.split_count() > 1
    {
        return Err(fail("unsupported format/version/recipe/alignment"));
    }
    let mut keys = HashSet::new();
    for entry in g.kv_entries() {
        let key = g.key_bytes(entry);
        if !keys.insert(key) || key.starts_with(b"glm_vq.") || key.starts_with(b"split.") {
            return Err(fail("duplicate/foreign metadata"));
        }
    }
    Ok(())
}

/// Whether this part opts into the Ling validator (and out of GLM's).
pub(super) fn requested(g: &GgufFile) -> bool {
    g.get_string("general.architecture") == Some(ARCH)
        || g.kv_entries()
            .iter()
            .any(|e| g.key_bytes(e).starts_with(b"ling_vq."))
}

/// One independent part: metadata, sizes, geometry, links and padding.
pub(super) fn validate_part(
    g: &GgufFile,
    tensors: &mut [TensorInfo],
    dir_end: usize,
    data_pos: u64,
) -> Result<()> {
    part_metadata(g)?;
    let mut names = HashSet::new();
    let mut headers = Vec::new();
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
            BF16 => t.elements.checked_mul(2).ok_or(TensorError::Overflow)?,
            FP8 | CODEBOOK => t.elements,
            Q4_K => {
                if t.ndim != 2 || t.dim[0] % Q4_BLOCK != 0 {
                    return Err(fail("unsupported Q4_K shape"));
                }
                t.elements
                    .checked_div(Q4_BLOCK)
                    .and_then(|n| n.checked_mul(Q4_BYTES))
                    .ok_or(TensorError::Overflow)?
            }
            VQ => {
                let header = Header::parse(g.as_bytes(), t)?;
                let bytes = header.size;
                headers.push((i, header));
                bytes
            }
            _ => return Err(fail("unsupported tensor type")),
        };
        let bytes = raw(g.as_bytes(), t)?;
        if t.typ == Q4_K && !STREAMED.with(Cell::get) {
            for block in bytes.chunks_exact(Q4_BYTES as usize) {
                if !q4_scales_finite(block) {
                    return Err(fail("nonfinite Q4_K scale"));
                }
            }
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
    validate_books(g, tensors, &headers)?;
    validate_native(g, tensors)
}
