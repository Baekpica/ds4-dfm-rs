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
pub use release::Gemma4VqFeature;

const ARCH: &[u8] = b"gemma4_vq";
const NAMESPACE: &[u8] = b"gemma4_vq.";
const VARIANT: &[u8] = b"gemma4-a4b-vq8";
const SOURCE_MODEL: &[u8] = b"gemma4";
const RECIPE: &[u8] = b"gemma-4-26b-a4b-family";
/// The R13+8 build of the same family recipe: the payload above plus a GVR8
/// residual tensor per expert matrix, container format version 2.
const RECIPE_R8: &[u8] = b"gemma-4-26b-a4b-family-r8";
const ALIGN: u64 = 128;
const HEADER: u64 = 128;
const VECTOR: u8 = 8;
const WORDS: u32 = 8192;
const FORMAT_VERSION: u32 = 1;
const RESIDUAL_VERSION: u32 = 2;
const RESIDUAL_BITS: u8 = 8;
const RESIDUAL_WORDS: u32 = 256;
const E4M3: u32 = 1;
const CODEBOOK: u32 = 0x4756_4342;
const VQ: u32 = 0x4756_5138;
const RESIDUAL: u32 = 0x4756_5238;
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
    feature: Gemma4VqFeature,
) -> Result<crate::tensors::TensorInventory> {
    release::open_dir(path, feature)
}

/// A directory whose `release.json` names this format.
pub(super) fn is_release(path: &std::path::Path) -> bool {
    release::is_release(path)
}

/// Device-resident bytes when the routed experts of the first `cold_layers`
/// layers stay host-cold: the DS4_GEMMA4_MOE_CPU_LAYERS and
/// DS4_GEMMA4_MOE_CACHE_MIB lanes, where the 4070 keeps the whole expert set
/// in the mmap and promotes the rest.
///
/// Term for term the native plan: `gemma4_stream_spans`
/// (ds4_gemma4_stream.inc:574-615) leaves every cold record out of the
/// resident spans and every other tensor in, and the cold records keep one
/// GVQ header each on the device (ds4: "Gemma-4 SSD cold records: 23040
/// headers, 2.81 MiB device"). The merged spans measure 2.345 GiB on the
/// R13+8 release, which is the engine's own "2.35 GiB device-resident".
///
/// `cold_layers` is the native count: all 30 for the slot-cache lane (the
/// engine forces all cold), N for the per-layer host split, 0 for the
/// resident lane. `None` when the artifact cannot be read, so the caller
/// falls back to the whole mapping.
pub fn resident_span_bytes(path: &std::path::Path, cold_layers: u32) -> Option<u64> {
    let cold_layers = cold_layers.min(30);
    let tensors = if path.is_dir() {
        if !is_release(path) {
            return None;
        }
        release::tensor_directory(path).ok()?
    } else {
        let g = GgufFile::open(path).ok()?;
        crate::tensors::TensorInventory::from_file(path, &g)
            .ok()?
            .tensors
    };
    Some(resident_span_of(&tensors, cold_layers))
}

/// The layer whose routed experts a tensor belongs to, `None` for every
/// other tensor. The native expert set is exactly `l->gate[e]`, `up[e]`,
/// `down[e]` and their R13+8 residual planes at
/// `model.layers.<il>.experts.<e>.<matrix>[.res]`
/// (ds4_gemma4_stream.inc:583-599). The shared per-stack books
/// (`...experts.<stack>.vq_codebook`, `...vq_residual_book`) name no expert
/// index and stay resident.
pub fn expert_layer(name: &str) -> Option<u32> {
    let fields: Vec<&str> = name.split('.').collect();
    if fields.len() < 6
        || fields[0] != "model"
        || fields[1] != "layers"
        || fields[3] != "experts"
        || fields[4].is_empty()
        || !fields[4].bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    fields[2].parse::<u32>().ok()
}

/// The bytes `--ssd-streaming` keeps resident: every tensor but the routed
/// expert records the slot cache streams (ds4_gemma4_stream.inc).
pub(crate) fn ssd_resident_bytes(tensors: &[crate::tensors::TensorInfo]) -> u64 {
    tensors
        .iter()
        .filter(|t| expert_layer(&t.name).is_none())
        .fold(0u64, |total, t| total.saturating_add(t.bytes))
}

fn resident_span_of(tensors: &[crate::tensors::TensorInfo], cold_layers: u32) -> u64 {
    tensors.iter().fold(0u64, |total, t| {
        let cold = expert_layer(&t.name).is_some_and(|il| il < cold_layers);
        total.saturating_add(if cold { HEADER } else { t.bytes })
    })
}

pub(super) fn identify(
    path: &std::path::Path,
) -> std::result::Result<crate::Identified, crate::IdentifyError> {
    model::identify(path)
}

pub(super) fn layouts() -> Vec<crate::LayoutSpec> {
    layout::layouts()
}

/// The drafter part's tensor set (`gemma4-vq8-mtp.gguf`): the 4-layer
/// assistant at hidden 1024 and the two projections across the backbone
/// width. The host binds `--mtp` against this plan.
pub(super) fn mtp_layouts() -> Vec<crate::LayoutSpec> {
    layout::mtp_layouts()
}

/// Bytes of the device-resident parts (the embedding stays on SSD; the MTP
/// part only with `Gemma4VqFeature::Mtp`).
pub(crate) fn resident_bytes(root: &std::path::Path, feature: Gemma4VqFeature) -> Result<u64> {
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
    feature: Gemma4VqFeature,
) -> std::result::Result<(), crate::ValidateError> {
    // The R13+8 build carries the residual tensors; the plain build does not,
    // and the inventory itself decides which contract applies.
    let residual = inventory.tensors.iter().any(|t| t.typ == RESIDUAL);
    let mut specs = layout::layouts_with(residual);
    if feature == Gemma4VqFeature::Mtp {
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

/// GVR8: one index byte per 8-weight vector into a K=256 E4M3 book; the
/// parent GVQ1 matrix supplies the row gain. 128-byte header, payload digest
/// at 96..128, canonical offsets (the converter's GVR1 writer).
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
        let payload = span(data, tensor.abs_offset, size)?;
        if Sha256::digest(&payload[HEADER as usize..end as usize])[..] != h[96..128] {
            return Err(fail("GVR1 payload checksum"));
        }
        zeros(payload, end, size)?;
        Ok(Self {
            book: u32_at(h, 24),
            size,
        })
    }
}

/// `gemma4_vq.residuals` maps each GVQ1 expert matrix to its GVR8 tensor. The
/// stage is all-or-none: every 13-bit matrix has exactly one residual and
/// every residual one parent (ds4-core/src/glm_vq.rs has the same rule).
fn validate_residuals(g: &GgufFile, tensors: &[TensorInfo]) -> Result<()> {
    let map = object(g, "gemma4_vq.residuals")?;
    let residuals = tensors.iter().filter(|t| t.typ == RESIDUAL).count();
    let matrices = tensors.iter().filter(|t| t.typ == VQ).count();
    if residuals == 0 && map.is_empty() {
        return Ok(());
    }
    if g.get_u32("gemma4_vq.format_version") != Some(RESIDUAL_VERSION) {
        return Err(fail("residual stage needs container version 2"));
    }
    let by_name: HashMap<_, _> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut linked = HashSet::new();
    for (parent, value) in &map {
        let name = value.as_str().ok_or(fail("invalid residual link"))?;
        let (Some(p), Some(r)) = (by_name.get(parent.as_str()), by_name.get(name)) else {
            return Err(fail("missing residual parent/tensor"));
        };
        if p.typ != VQ
            || r.typ != RESIDUAL
            || p.dim[..2] != r.dim[..2]
            || !linked.insert(r.name.as_str())
        {
            return Err(fail("residual link type/shape/duplicate"));
        }
    }
    if linked.len() != residuals || residuals != matrices {
        return Err(fail("partial or unlinked residual stage"));
    }
    Ok(())
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
        // A GVQ1 header carries the book digest at 64..96; a GVR1 header does
        // not (its own payload digest at 96..128 was checked when the header
        // was parsed), so only the GVQ1 link compares the book here.
        if tensors[link.tensor].typ == VQ
            && raw(g.as_bytes(), &tensors[link.tensor])?[64..96] != hash
        {
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
    // Version 1 is the plain 13-bit artifact; version 2 adds the R13+8
    // residual tensors and carries the -r8 recipe id. Either is accepted, and
    // the version decides which recipe id must accompany it.
    let supported = match g.get_u32("gemma4_vq.format_version") {
        Some(FORMAT_VERSION) => g.get_string("gemma4_vq.recipe_id") == Some(RECIPE),
        Some(RESIDUAL_VERSION) => g.get_string("gemma4_vq.recipe_id") == Some(RECIPE_R8),
        _ => false,
    };
    if g.get_string("general.architecture") != Some(ARCH)
        || g.alignment != ALIGN
        || !supported
        || g.get_string("gemma4_vq.variant") != Some(VARIANT)
        || g.get_string("gemma4_vq.source_model_type") != Some(SOURCE_MODEL)
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
        // `tokenizer.*` is GGUF's own namespace, not a foreign key: the trunk
        // part carries the vocabulary, the merges and the chat template there.
        if !keys.insert(key)
            || !(key.starts_with(b"general.")
                || key.starts_with(NAMESPACE)
                || key.starts_with(b"tokenizer."))
        {
            return Err(fail("duplicate/foreign metadata"));
        }
    }
    Ok(())
}

/// Payload bytes of a block-quantized matrix: GGUF dim 0 is the input width.
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
        match t.typ {
            Q8_0 if !bytes
                .as_chunks::<{ Q8_BYTES as usize }>()
                .0
                .iter()
                .all(|block| q8_scale_finite(block)) =>
            {
                return Err(fail("nonfinite Q8_0 scale"));
            }
            Q4_K if !bytes
                .as_chunks::<{ Q4_BYTES as usize }>()
                .0
                .iter()
                .all(|block| q4_scales_finite(block)) =>
            {
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
    validate_books(g, tensors, &links)?;
    validate_residuals(g, tensors)
}

#[cfg(test)]
mod span_tests {
    use super::*;
    use crate::tensors::TensorInfo;

    fn tensor(name: &str, bytes: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            ndim: 2,
            dim: [0; crate::tensors::MAX_DIMS as usize],
            typ: VQ,
            rel_offset: 0,
            abs_offset: 0,
            elements: 0,
            bytes,
            shard: 0,
        }
    }

    #[test]
    fn expert_records_are_the_per_expert_matrices_only() {
        // ds4_gemma4_stream.inc:583-599 marks l->gate[e], up[e], down[e] and
        // the R13+8 residual planes; the shared per-stack books stay resident.
        assert_eq!(expert_layer("model.layers.0.experts.0.down_proj"), Some(0));
        assert_eq!(
            expert_layer("model.layers.29.experts.127.gate_proj"),
            Some(29)
        );
        assert_eq!(
            expert_layer("model.layers.7.experts.12.up_proj.res"),
            Some(7)
        );
        assert_eq!(
            expert_layer("model.layers.0.experts.down_proj.vq_codebook"),
            None
        );
        assert_eq!(
            expert_layer("model.layers.0.experts.gate_up_proj.vq_residual_book"),
            None
        );
        assert_eq!(expert_layer("model.layers.0.self_attn.q_proj.weight"), None);
        assert_eq!(expert_layer("model.embed_tokens.weight"), None);
    }

    #[test]
    fn cold_records_leave_one_header_each() {
        let tensors = vec![
            tensor("model.layers.0.experts.0.gate_proj", 1_000),
            tensor("model.layers.0.experts.0.down_proj.res", 500),
            tensor("model.layers.0.experts.down_proj.vq_codebook", 70_000),
            tensor("model.layers.0.mlp.gate.weight", 3_000),
            tensor("model.layers.1.experts.0.gate_proj", 1_000),
            tensor("model.embed_tokens.weight", 9_000),
        ];
        // Nothing cold: every tensor whole.
        assert_eq!(
            resident_span_of(&tensors, 0),
            1_000 + 500 + 70_000 + 3_000 + 1_000 + 9_000
        );
        // Layer 0 cold: its two expert records leave one 128 B header each.
        assert_eq!(
            resident_span_of(&tensors, 1),
            HEADER + HEADER + 70_000 + 3_000 + 1_000 + 9_000
        );
        // Every layer cold: both records of layer 0 and layer 1 are headers.
        assert_eq!(
            resident_span_of(&tensors, 30),
            HEADER * 3 + 70_000 + 3_000 + 9_000
        );
    }
}
