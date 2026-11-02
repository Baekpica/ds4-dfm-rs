//! Gemma-4 26B A4B VQ (`gemma4_vq`) is its own artifact contract, in the shape
//! every VQ family in this tree uses: a release.json directory of GGUF parts
//! whose routed experts are GVQ1 matrices linked to shared GVBC codebook
//! tensors, under a per-family metadata namespace
//! (docs/ds4-dfm-model-families.md "Design contract").
//!
//! The artifact this module is written from is the family release the toolkit
//! emits: a trunk part (11520 GVQ1 matrices over 60 K8192 books, plus 175
//! Q4_K, 30 Q8_0 and 391 BF16 tensors) and an embedding part (the single BF16
//! token table the host reads by row). release.json names both, files.json
//! hashes every file including the source config.json, and each part carries
//! this family's metadata.
//!
//! Deviation, named: the artifact records no source revision, so a part is
//! bound by its SHA-256 and the checkpoint config in files.json rather than by
//! an upstream commit id, which MiMo's contract additionally requires.

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
pub use release::Feature;

const ARCH: &[u8] = b"gemma4_vq";
const NAMESPACE: &[u8] = b"gemma4_vq.";
const VARIANT: &[u8] = b"gemma4-a4b-vq8";
const SOURCE_MODEL: &[u8] = b"gemma4";
const RECIPE: &[u8] = b"gemma-4-26b-a4b-family";
const ALIGN: u64 = 128;
const HEADER: u64 = 128;
const VECTOR: u8 = 8;
const WORDS: u32 = 8192;
const FORMAT_VERSION: u32 = 1;
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

/// The device-copied parts (trunk, and the MTP part on request), then the
/// embedding part, whose rows the host reads per token, last.
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

/// Bytes of the device-resident parts (the embedding stays on SSD; the MTP
/// part only with `Feature::Mtp`).
pub(crate) fn resident_bytes(root: &std::path::Path, feature: Feature) -> Result<u64> {
    release::resident_bytes(root, feature)
}

pub(super) fn metadata(root: &std::path::Path) -> Result<GgufFile> {
    Ok(GgufFile::open(&release::first_part(root)?)?)
}

/// A part this family owns: its architecture string, or any metadata under
/// this namespace, whichever is present. The tensor parser dispatches on it.
pub(super) fn requested(g: &GgufFile) -> bool {
    g.get_string("general.architecture") == Some(ARCH)
        || g.kv_entries()
            .iter()
            .any(|e| g.key_bytes(e).starts_with(NAMESPACE))
}

/// Every inventory tensor must be the layout's tensor, with the type and the
/// dimensions the artifact stores. This is the check the family's bind plan
/// would run; the layout is the contract, so it is compared directly. The MTP
/// drafter is its own part with its own tensor set.
pub(super) fn validate_inventory(
    inventory: &crate::TensorInventory,
    feature: Feature,
) -> std::result::Result<(), crate::ValidateError> {
    let mut specs = layout::layouts();
    if feature == Feature::Mtp {
        specs.extend(layout::mtp_layouts());
    }
    let by_name: HashMap<&str, &crate::LayoutSpec> =
        specs.iter().map(|s| (s.name.as_str(), s)).collect();
    let mut seen = HashSet::new();
    for tensor in &inventory.tensors {
        let Some(spec) = by_name.get(tensor.name.as_str()) else {
            return Err(crate::ValidateError::Token(
                "Gemma-4 VQ tensor set mismatch",
            ));
        };
        let crate::TypeClass::Exact(typ) = spec.class else {
            return Err(crate::ValidateError::Token("Gemma-4 VQ tensor class"));
        };
        if !seen.insert(tensor.name.as_str())
            || tensor.typ != typ
            || tensor.ndim != spec.ndim
            || tensor.dim[..tensor.ndim as usize] != spec.dim[..spec.ndim as usize]
        {
            return Err(crate::ValidateError::Token("Gemma-4 VQ tensor layout"));
        }
    }
    if seen.len() != specs.len() {
        return Err(crate::ValidateError::Token(
            "Gemma-4 VQ tensor set mismatch",
        ));
    }
    Ok(())
}

fn fail(reason: &'static str) -> TensorError {
    TensorError::Gemma4Vq(reason)
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
    let books = object(g, "gemma4_vq.codebooks")?;
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

fn part_metadata(g: &GgufFile) -> Result<()> {
    if g.get_string("general.architecture") != Some(ARCH)
        || g.alignment != ALIGN
        || g.get_u32("gemma4_vq.format_version") != Some(FORMAT_VERSION)
        || g.get_string("gemma4_vq.variant") != Some(VARIANT)
        || g.get_string("gemma4_vq.source_model_type") != Some(SOURCE_MODEL)
        || g.get_string("gemma4_vq.recipe_id") != Some(RECIPE)
        || !matches!(
            g.get_string("gemma4_vq.part_role"),
            Some(b"trunk") | Some(b"embedding") | Some(b"mtp")
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
            BF16 => t.elements.checked_mul(2).ok_or(TensorError::Overflow)?,
            CODEBOOK => t.elements,
            Q8_0 => quant_bytes(t, Q8_BLOCK, Q8_BYTES)?,
            Q4_K => quant_bytes(t, Q4_BLOCK, Q4_BYTES)?,
            VQ => {
                let header = Header::parse(g.as_bytes(), t)?;
                links.push(BookLink {
                    tensor: i,
                    book: header.book,
                    words: WORDS,
                });
                header.size
            }
            _ => return Err(fail("unsupported tensor type")),
        };
        let bytes = raw(g.as_bytes(), t)?;
        if STREAMED.with(Cell::get) {
            continue;
        }
        match t.typ {
            Q8_0 if !bytes.chunks_exact(Q8_BYTES as usize).all(q8_scale_finite) => {
                return Err(fail("nonfinite Q8_0 scale"));
            }
            Q4_K if !bytes.chunks_exact(Q4_BYTES as usize).all(q4_scales_finite) => {
                return Err(fail("nonfinite Q4_K scale"));
            }
            _ => {}
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
