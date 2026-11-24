//! K2-Horizon VQ (`k2_horizon_vq`) is its own artifact contract.
//! Byte layout: the release's FORMAT.md (`k2-horizon-vq8`, format version
//! 1). Routed experts are 13-bit GVQ1 matrices (K = 8192, one E4M3 book per
//! layer, F16 row gains, a 12-bit pair stream and a 13th-bit plane, as
//! MiniMax's). Besides the trunk (58 expert parts and the body) the release
//! has an `embedding` part (the Q8_0 token table, uploaded whole or read in
//! host rows). Validate borrowed mmap spans; never expand experts or copy a
//! part.

mod layout;
mod model;
mod release;

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::gguf::GgufFile;
use crate::tensors::{TensorError, TensorInfo};

const ARCH: &[u8] = b"k2_horizon_vq";
const NAMESPACE: &[u8] = b"k2_horizon_vq.";
const VARIANT: &[u8] = b"k2-horizon-vq8";
const SOURCE_MODEL: &[u8] = b"k2_horizon";
const RECIPE: &[u8] = b"k2-horizon-hvq8-r13x58-q8-nvfp4kv-v1";
const ALIGN: u64 = 128;
const HEADER: u64 = 128;
const VECTOR: u8 = 8;
const INDEX_BITS: u8 = 13;
const WORDS: u32 = 8192;
const FORMAT_VERSION: u32 = 1;
const E4M3: u32 = 1;
const CODEBOOK: u32 = 0x4756_4342;
const VQ: u32 = 0x4756_5138;
const F32: u32 = 0;
const Q8_0: u32 = 8;
const Q8_BLOCK: u64 = 32;
const Q8_BYTES: u64 = 34;

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

/// The f16 scale at the start of a Q8_0 block is finite.
fn q8_scale_finite(scale: &[u8]) -> bool {
    u16::from_le_bytes([scale[0], scale[1]]) & 0x7c00 != 0x7c00
}

/// The trunk parts in release order, then the embedding part, whose table
/// native finds as the inventory's last tensor.
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

/// A pinned source file kept beside the parts (release.rs ASSETS).
pub(super) fn asset(root: &std::path::Path, name: &str) -> Result<Vec<u8>> {
    release::asset(root, name)
}

/// Bytes of the trunk parts the device copy holds (the embedding part is
/// read in host rows unless native uploads it).
pub(super) fn trunk_bytes(root: &std::path::Path) -> Result<u64> {
    release::trunk_bytes(root)
}

pub(super) fn metadata(root: &std::path::Path) -> Result<GgufFile> {
    Ok(GgufFile::open(&release::first_part(root)?)?)
}

pub(super) fn validate_inventory(
    inventory: &crate::TensorInventory,
) -> std::result::Result<(), crate::ValidateError> {
    let plan = crate::BindPlan::resolve(crate::shape::SHAPE_K2H_VQ, inventory);
    let found = plan.slots.iter().filter(|s| s.tensor.is_some()).count();
    if found != inventory.tensors.len() || !plan.missing_required().is_empty() {
        return Err(crate::ValidateError::Token(
            "K2-Horizon VQ tensor set mismatch",
        ));
    }
    crate::validate_layouts(&plan)
        .map_err(|e| crate::ValidateError::TokenKey("K2-Horizon VQ tensor layout", e.to_string()))
}

fn fail(reason: &'static str) -> TensorError {
    TensorError::K2hVq(reason)
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

/// GVQ1 matrix header (FORMAT.md "GVQ1 matrix"): 13-bit indices, the low 12
/// bits as a pair stream, bit 12 in a plane with its own row stride.
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
            || h[6] != INDEX_BITS
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
        let plane = align(main_end)?;
        let end = plane.checked_add(elems / 64).ok_or(TensorError::Overflow)?;
        let size = align(end)?;
        if [u64_at(h, 32), u64_at(h, 40), u64_at(h, 48), u64_at(h, 56)]
            != [HEADER, main, plane, size]
        {
            return Err(fail("noncanonical GVQ1 offsets"));
        }
        let payload = span(data, tensor.abs_offset, size)?;
        for gain in span(payload, HEADER, 2 * rows)?.as_chunks::<2>().0 {
            let bits = u16::from_le_bytes(*gain);
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

/// A payload's codebook reference: tensor index and book ID.
struct BookLink {
    tensor: usize,
    book: u32,
}

fn object(g: &GgufFile, key: &str) -> Result<Map<String, Value>> {
    let Some(bytes) = g.get_string(key) else {
        return Err(fail("missing codebook map"));
    };
    match serde_json::from_slice(bytes) {
        Ok(Value::Object(map)) => Ok(map),
        _ => Err(fail("invalid codebook map")),
    }
}

fn raw<'a>(data: &'a [u8], tensor: &TensorInfo) -> Result<&'a [u8]> {
    span(data, tensor.abs_offset, tensor.bytes)
}

/// Every expert projection of layer L links the part's one book, ID 4L
/// (FORMAT.md "Expert parts"; 4L + 1 is reserved for a residual stage).
fn book_allowed(name: &str, layer: u32, id: u32) -> bool {
    let prefix = format!("model.layers.{layer}.mlp.experts.");
    let projection = name.strip_prefix(&prefix).is_some_and(|rest| {
        [".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"]
            .iter()
            .any(|p| rest.ends_with(p))
    });
    projection && layer.checked_mul(4) == Some(id)
}

/// Every GVQ1 matrix names its layer's book by ID; the map lists exactly
/// the books matrices use, and each matrix carries its book's SHA-256. The
/// body and embedding parts link no book.
fn validate_books(g: &GgufFile, tensors: &[TensorInfo], links: &[BookLink]) -> Result<()> {
    let books = object(g, "k2_horizon_vq.codebooks")?;
    let layer = g.get_u32("k2_horizon_vq.layer");
    let bits = g.get_u32("k2_horizon_vq.index_bits");
    if links.is_empty() {
        return if books.is_empty() && layer.is_none() && bits.is_none() {
            Ok(())
        } else {
            Err(fail("book metadata outside an expert part"))
        };
    }
    let Some(layer) = layer.filter(|_| bits == Some(u32::from(INDEX_BITS))) else {
        return Err(fail("expert part without layer/13-bit indices"));
    };
    let by_name: HashMap<_, _> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut used = HashSet::new();
    let mut hashes = HashMap::new();
    for link in links {
        let tensor = &tensors[link.tensor];
        if !book_allowed(&tensor.name, layer, link.book) {
            return Err(fail("matrix book outside the part's plan"));
        }
        let id = link.book.to_string();
        let name = books
            .get(&id)
            .and_then(Value::as_str)
            .ok_or(fail("missing codebook link"))?;
        let book = by_name.get(name).ok_or(fail("missing codebook tensor"))?;
        if book.typ != CODEBOOK
            || book.ndim != 2
            || book.dim[..2] != [u64::from(VECTOR), u64::from(WORDS)]
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
        // release.json's part SHA-256 already covers every payload byte.
        if raw(g.as_bytes(), tensor)?[64..96] != hash {
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

fn part_metadata(g: &GgufFile) -> Result<()> {
    if g.get_string("general.architecture") != Some(ARCH)
        || g.alignment != ALIGN
        || g.get_u32("k2_horizon_vq.format_version") != Some(FORMAT_VERSION)
        || g.get_string("k2_horizon_vq.variant") != Some(VARIANT)
        || g.get_string("k2_horizon_vq.source_model_type") != Some(SOURCE_MODEL)
        || g.get_string("k2_horizon_vq.recipe_id") != Some(RECIPE)
        || !matches!(
            g.get_string("k2_horizon_vq.part_role"),
            Some(b"trunk") | Some(b"embedding")
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
    Ok(())
}

/// Whether this part opts into the K2-Horizon validator (checked before the
/// other GVQ1 families', which would claim its tensors).
pub(super) fn requested(g: &GgufFile) -> bool {
    g.get_string("general.architecture") == Some(ARCH)
        || g.kv_entries()
            .iter()
            .any(|e| g.key_bytes(e).starts_with(NAMESPACE))
}

/// Payload bytes of a Q8_0 matrix: GGUF dim 0 is the input width.
fn q8_bytes(t: &TensorInfo) -> Result<u64> {
    if t.ndim != 2 || !t.dim[0].is_multiple_of(Q8_BLOCK) {
        return Err(fail("unsupported quant shape"));
    }
    t.elements
        .checked_div(Q8_BLOCK)
        .and_then(|n| n.checked_mul(Q8_BYTES))
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
            CODEBOOK => t.elements,
            Q8_0 => q8_bytes(t)?,
            VQ => {
                let header = Header::parse(g.as_bytes(), t)?;
                links.push(BookLink {
                    tensor: i,
                    book: header.book,
                });
                header.size
            }
            _ => return Err(fail("unsupported tensor type")),
        };
        let bytes = raw(g.as_bytes(), t)?;
        if STREAMED.with(Cell::get) || t.typ != Q8_0 {
            continue;
        }
        if !bytes
            .as_chunks::<{ Q8_BYTES as usize }>()
            .0
            .iter()
            .all(|b| q8_scale_finite(b))
        {
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
    validate_books(g, tensors, &links)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projections_link_only_their_layer_book() {
        let gate = "model.layers.16.mlp.experts.3.gate_proj.weight";
        let up = "model.layers.16.mlp.experts.191.up_proj.weight";
        let down = "model.layers.16.mlp.experts.0.down_proj.weight";
        for name in [gate, up, down] {
            assert!(book_allowed(name, 16, 64), "{name}");
            // 4L + 1 is the reserved residual slot; R13 links none.
            assert!(!book_allowed(name, 16, 65), "{name}");
            assert!(!book_allowed(name, 15, 60), "{name}");
        }
        assert!(!book_allowed("model.layers.16.mlp.vq_codebook", 16, 64));
        assert!(!book_allowed(
            "model.layers.16.mlp.shared_experts.gate_proj.weight",
            16,
            64
        ));
        // 4L must not wrap.
        let far = format!("model.layers.{}.mlp.experts.0.gate_proj.weight", u32::MAX);
        assert!(!book_allowed(&far, u32::MAX, u32::MAX.wrapping_mul(4)));
    }
}
