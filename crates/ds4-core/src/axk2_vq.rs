//! A.X-K2 VQ (`axk2_vq`, GVQ1 version 2) is its own artifact contract.
//! Byte layout: Baekpica/A.X-K2-VQ-GGUF FORMAT.md at
//! a0c3b02e9b45dc67ef8dd1d48eb48034c93d6607. Routed experts store 9- or
//! 10-bit indices byte-major (one low byte per index, then a plane with the
//! high bit or two); each layer has gate (4L), down (4L+2) and up (4L+3)
//! books. The release adds Q8_0 indexer projections and an `embedding`
//! part, the BF16 token table the host reads in rows and never uploads.
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

pub use layout::EMBEDDING;

const ARCH: &[u8] = b"axk2_vq";
const VARIANT: &[u8] = b"axk2-vq8";
const SOURCE_MODEL: &[u8] = b"axk2";
const RECIPE: &[u8] = b"axk2-vq8-r10x7-r9x53-wlloyd-q4k-q8idx-embssd-v1";
const ALIGN: u64 = 128;
const HEADER: u64 = 128;
const VECTOR: u8 = 8;
const FORMAT_VERSION: u32 = 1;
const GVQ_VERSION: u16 = 2;
const E4M3: u32 = 1;
const CODEBOOK: u32 = 0x4756_4342;
const VQ: u32 = 0x4756_5138;
const F32: u32 = 0;
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
        .as_chunks::<2>()
        .0
        .iter()
        .all(|s| u16::from_le_bytes(*s) & 0x7c00 != 0x7c00)
}

/// The f16 scale at the start of a Q8_0 block is finite.
fn q8_scale_finite(scale: &[u8]) -> bool {
    u16::from_le_bytes([scale[0], scale[1]]) & 0x7c00 != 0x7c00
}

/// Every part: the trunk, which the device copy holds, and the embedding
/// part, whose rows the host reads per token.
pub(super) fn open_dir(path: &std::path::Path) -> Result<crate::tensors::TensorInventory> {
    release::open_dir(path)
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

/// Bytes of the device-resident trunk parts (the embedding part stays on SSD).
pub(super) fn trunk_bytes(root: &std::path::Path) -> Result<u64> {
    release::trunk_bytes(root)
}

pub(super) fn metadata(root: &std::path::Path) -> Result<GgufFile> {
    Ok(GgufFile::open(&release::first_part(root)?)?)
}

pub(super) fn validate_inventory(
    inventory: &crate::TensorInventory,
) -> std::result::Result<(), crate::ValidateError> {
    let plan = crate::BindPlan::resolve(crate::shape::SHAPE_AXK2_VQ, inventory);
    let found = plan.slots.iter().filter(|s| s.tensor.is_some()).count();
    if found != inventory.tensors.len() || !plan.missing_required().is_empty() {
        return Err(crate::ValidateError::Token("A.X-K2 VQ tensor set mismatch"));
    }
    crate::validate_layouts(&plan)
        .map_err(|e| crate::ValidateError::TokenKey("A.X-K2 VQ tensor layout", e.to_string()))
}

fn fail(reason: &'static str) -> TensorError {
    TensorError::Axk2Vq(reason)
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

/// GVQ1 version 2 matrix header (FORMAT.md "GVQ1 matrix, version 2"): a
/// byte per index (its low 8 bits), then a plane of the high `bits - 8`
/// bits. K512 books carry 9-bit indices, K1024 books 10-bit ones.
struct Header {
    bits: u8,
    book: u32,
    size: u64,
}

impl Header {
    fn parse(data: &[u8], tensor: &TensorInfo) -> Result<Self> {
        let h = span(data, tensor.abs_offset, HEADER)?;
        let rows = u64::from(u32_at(h, 8));
        let cols = u64::from(u32_at(h, 12));
        let bits = h[6];
        let k = u32_at(h, 16);
        if &h[..4] != b"GVQ1"
            || u16::from_le_bytes([h[4], h[5]]) != GVQ_VERSION
            || h[7] != VECTOR
            || !matches!((bits, k), (9, 512) | (10, 1024))
            || u32_at(h, 20) != E4M3
            || u32_at(h, 28) != 0
            || rows == 0
            || cols == 0
            || cols % 64 != 0
            || tensor.ndim != 2
            || tensor.dim[..2] != [cols, rows]
        {
            return Err(fail("invalid GVQ1 v2 header/shape"));
        }
        let vectors = rows.checked_mul(cols / 8).ok_or(TensorError::Overflow)?;
        let main = align(HEADER + 2 * rows)?;
        let plane = align(main.checked_add(vectors).ok_or(TensorError::Overflow)?)?;
        let end = plane
            .checked_add(vectors * u64::from(bits - 8) / 8)
            .ok_or(TensorError::Overflow)?;
        let size = align(end)?;
        if [u64_at(h, 32), u64_at(h, 40), u64_at(h, 48), u64_at(h, 56)]
            != [HEADER, main, plane, size]
        {
            return Err(fail("noncanonical GVQ1 offsets"));
        }
        let payload = span(data, tensor.abs_offset, size)?;
        for gain in span(payload, HEADER, 2 * rows)?.as_chunks::<2>().0 {
            let half = u16::from_le_bytes(*gain);
            // +0 is the zero-row encoding; -0 and negatives are invalid.
            if half & 0x8000 != 0 || half & 0x7c00 == 0x7c00 {
                return Err(fail("negative/nonfinite row gain"));
            }
        }
        zeros(payload, HEADER + 2 * rows, main)?;
        zeros(payload, main + vectors, plane)?;
        zeros(payload, end, size)?;
        Ok(Self {
            bits,
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

/// The books a projection may link (FORMAT.md "Expert parts"): gate the
/// layer book 4L; up 4L or its own 4L+3; down 4L or its own 4L+2.
fn book_allowed(name: &str, layer: u32, id: u32) -> bool {
    let base = 4 * layer;
    if name.ends_with(".gate_proj.weight") {
        id == base
    } else if name.ends_with(".up_proj.weight") {
        id == base || id == base + 3
    } else if name.ends_with(".down_proj.weight") {
        id == base || id == base + 2
    } else {
        false
    }
}

fn validate_books(g: &GgufFile, tensors: &[TensorInfo], headers: &[(usize, Header)]) -> Result<()> {
    let links = object(g, "axk2_vq.codebooks")?;
    let layer = g.get_u32("axk2_vq.layer");
    let bits = g.get_u32("axk2_vq.index_bits");
    if headers.is_empty() {
        // The body and embedding parts link no book.
        return if links.is_empty() && layer.is_none() && bits.is_none() {
            Ok(())
        } else {
            Err(fail("book metadata outside an expert part"))
        };
    }
    let (Some(layer), Some(bits)) = (layer, bits) else {
        return Err(fail("expert part without layer/index bits"));
    };
    let by_name: HashMap<_, _> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut used = HashSet::new();
    let mut hashes = HashMap::new();
    for (i, header) in headers {
        let tensor = &tensors[*i];
        if u32::from(header.bits) != bits || !book_allowed(&tensor.name, layer, header.book) {
            return Err(fail("matrix bits/book outside the part's plan"));
        }
        let id = header.book.to_string();
        let name = links
            .get(&id)
            .and_then(Value::as_str)
            .ok_or(fail("missing codebook link"))?;
        let book = by_name.get(name).ok_or(fail("missing codebook tensor"))?;
        if book.typ != CODEBOOK
            || book.ndim != 2
            || book.dim[..2] != [u64::from(VECTOR), 1u64 << header.bits]
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
        if raw(g.as_bytes(), tensor)?[64..96] != hash {
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

fn part_metadata(g: &GgufFile) -> Result<()> {
    if g.get_string("general.architecture") != Some(ARCH)
        || g.alignment != ALIGN
        || g.get_u32("axk2_vq.format_version") != Some(FORMAT_VERSION)
        || g.get_string("axk2_vq.variant") != Some(VARIANT)
        || g.get_string("axk2_vq.source_model_type") != Some(SOURCE_MODEL)
        || g.get_string("axk2_vq.recipe_id") != Some(RECIPE)
        || !matches!(
            g.get_string("axk2_vq.part_role"),
            Some(b"trunk") | Some(b"embedding")
        )
        || g.split_count() > 1
    {
        return Err(fail("unsupported format/version/recipe/alignment"));
    }
    let mut keys = HashSet::new();
    for entry in g.kv_entries() {
        let key = g.key_bytes(entry);
        if !keys.insert(key)
            || key.starts_with(b"glm_vq.")
            || key.starts_with(b"ling_vq.")
            || key.starts_with(b"beam_vq.")
            || key.starts_with(b"split.")
        {
            return Err(fail("duplicate/foreign metadata"));
        }
    }
    Ok(())
}

/// Whether this part opts into the A.X-K2 validator (checked before Ling's
/// and GLM's, which would claim its GVQ1 tensors).
pub(super) fn requested(g: &GgufFile) -> bool {
    g.get_string("general.architecture") == Some(ARCH)
        || g.kv_entries()
            .iter()
            .any(|e| g.key_bytes(e).starts_with(b"axk2_vq."))
}

fn quant_bytes(t: &TensorInfo, block: u64, bytes: u64) -> Result<u64> {
    if t.ndim != 2 || !t.dim[0].is_multiple_of(block) {
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
            CODEBOOK => t.elements,
            Q4_K => quant_bytes(t, Q4_BLOCK, Q4_BYTES)?,
            Q8_0 => quant_bytes(t, Q8_BLOCK, Q8_BYTES)?,
            VQ => {
                let header = Header::parse(g.as_bytes(), t)?;
                let bytes = header.size;
                headers.push((i, header));
                bytes
            }
            _ => return Err(fail("unsupported tensor type")),
        };
        if STREAMED.with(Cell::get) {
            continue;
        }
        let bytes = raw(g.as_bytes(), t)?;
        let (q4, _) = bytes.as_chunks::<{ Q4_BYTES as usize }>();
        if t.typ == Q4_K && !q4.iter().all(|b| q4_scales_finite(b)) {
            return Err(fail("nonfinite Q4_K scale"));
        }
        let (q8, _) = bytes.as_chunks::<{ Q8_BYTES as usize }>();
        if t.typ == Q8_0 && !q8.iter().all(|b| q8_scale_finite(b)) {
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
    validate_books(g, tensors, &headers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projections_link_only_their_books() {
        let gate = "model.layers.16.mlp.experts.3.gate_proj.weight";
        let up = "model.layers.16.mlp.experts.3.up_proj.weight";
        let down = "model.layers.16.mlp.experts.3.down_proj.weight";
        assert!(book_allowed(gate, 16, 64));
        assert!(!book_allowed(gate, 16, 67));
        assert!(book_allowed(up, 16, 67) && book_allowed(up, 16, 64));
        assert!(!book_allowed(up, 16, 66));
        assert!(book_allowed(down, 16, 66) && !book_allowed(down, 16, 67));
        assert!(!book_allowed(down, 15, 66));
        assert!(!book_allowed("model.layers.16.mlp.vq_codebook", 16, 64));
    }
}
