//! GLM GVQ1 v1 is a separate artifact contract from GGML/MXQ and DQVL.
//! Byte layout: Baekpica/GLM-5.3-Flash-VQ-GGUF, FORMAT.md at
//! fe2a446e774624ef8d397136d81524c61db77f65. Validate borrowed mmap spans;
//! never expand experts or copy an entire part while building the inventory.

mod release;

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::gguf::GgufFile;
use crate::tensors::{TensorError, TensorInfo};

const ALIGN: u64 = 128;
const HEADER: u64 = 128;
const VECTOR: u8 = 8;
const FORMAT_VERSION: u32 = 1;
const FP8: u32 = 0x4751_4638;
const CODEBOOK: u32 = 0x4756_4342;
const VQ: u32 = 0x4756_5138;
const F32: u32 = 0;
const F16: u32 = 1;
const BF16: u32 = 30;
const SCALE_BLOCK: u64 = 128;
const PATCH: &str = "model.visual.patch_embed.proj.weight";

type Result<T> = std::result::Result<T, TensorError>;

pub(super) fn open_dir(path: &std::path::Path) -> Result<crate::tensors::TensorInventory> {
    release::open_dir(path)
}

fn fail(reason: &'static str) -> TensorError {
    TensorError::GlmVq(reason)
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

struct Header {
    k: u32,
    precision: u32,
    book: u32,
    size: u64,
}

impl Header {
    fn parse(data: &[u8], tensor: &TensorInfo) -> Result<Self> {
        let h = span(data, tensor.abs_offset, HEADER)?;
        let rows = u64::from(u32_at(h, 8));
        let cols = u64::from(u32_at(h, 12));
        let k = u32_at(h, 16);
        let precision = u32_at(h, 20);
        if &h[..4] != b"GVQ1"
            || h[4..6] != [1, 0]
            || h[7] != VECTOR
            || !matches!(k, 4096 | 8192)
            || h[6] != if k == 4096 { 12 } else { 13 }
            || !matches!(precision, 1 | 2)
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
        if Sha256::digest(&payload[HEADER as usize..])[..] != h[96..128] {
            return Err(fail("GVQ1 payload checksum"));
        }
        for gain in span(payload, HEADER, 2 * rows)?.chunks_exact(2) {
            let bits = u16::from_le_bytes(gain.try_into().unwrap());
            if bits == 0 || bits & 0x8000 != 0 || bits & 0x7c00 == 0x7c00 {
                return Err(fail("nonpositive/nonfinite row gain"));
            }
        }
        zeros(payload, HEADER + 2 * rows, main)?;
        if plane != 0 {
            zeros(payload, main_end, plane)?;
        }
        zeros(payload, end, size)?;
        Ok(Self {
            k,
            precision,
            book: u32_at(h, 24),
            size,
        })
    }
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

fn book_hash(data: &[u8], book: &TensorInfo) -> Result<[u8; 32]> {
    let bytes = raw(data, book)?;
    let finite = match book.typ {
        CODEBOOK => bytes.iter().all(|x| x & 0x7f != 0x7f),
        F16 => bytes
            .chunks_exact(2)
            .all(|x| u16::from_le_bytes(x.try_into().unwrap()) & 0x7c00 != 0x7c00),
        _ => false,
    };
    if !finite {
        return Err(fail("invalid/nonfinite codebook"));
    }
    Ok(Sha256::digest(bytes).into())
}

fn validate_books(g: &GgufFile, tensors: &[TensorInfo], headers: &[(usize, Header)]) -> Result<()> {
    let links = object(g, "glm_vq.codebooks")?;
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
        let typ = if header.precision == 1 { CODEBOOK } else { F16 };
        if book.typ != typ
            || book.ndim != 2
            || book.dim[..2] != [u64::from(VECTOR), u64::from(header.k)]
        {
            return Err(fail("codebook type/shape"));
        }
        let hash = match hashes.get(name) {
            Some(hash) => *hash,
            None => {
                let hash = book_hash(g.as_bytes(), book)?;
                hashes.insert(name, hash);
                hash
            }
        };
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
    let scales = object(g, "glm_vq.fp8_scales")?;
    let mut seen = 0;
    for t in tensors.iter().filter(|t| t.typ == FP8) {
        let name = scales
            .get(&t.name)
            .and_then(Value::as_str)
            .ok_or(fail("missing FP8 scale link"))?;
        let scale = by_name.get(name).ok_or(fail("missing FP8 scale tensor"))?;
        if t.ndim != 2
            || t.dim[0] % SCALE_BLOCK != 0
            || t.dim[1] % SCALE_BLOCK != 0
            || scale.typ != F32
            || scale.ndim != 2
            || scale.dim[..2] != [t.dim[0] / SCALE_BLOCK, t.dim[1] / SCALE_BLOCK]
        {
            return Err(fail("FP8 scale type/shape"));
        }
        // weight_scale_inv is an original multiplicative F32 scale, not UE8M0.
        for bytes in raw(g.as_bytes(), scale)?.chunks_exact(4) {
            let value = f32::from_le_bytes(bytes.try_into().unwrap());
            if !value.is_finite() || value <= 0.0 {
                return Err(fail("nonpositive/nonfinite FP8 scale"));
            }
        }
        seen += 1;
    }
    if seen != scales.len() {
        return Err(fail("surplus FP8 scale links"));
    }
    for (name, shape) in object(g, "glm_vq.logical_shapes")? {
        let tensor = by_name
            .get(name.as_str())
            .ok_or(fail("missing logical-shape tensor"))?;
        if name != PATCH
            || shape != serde_json::json!([1024, 3, 2, 14, 14])
            || tensor.ndim != 2
            || tensor.dim[..2] != [1176, 1024]
            || !matches!(tensor.typ, F32 | F16 | BF16)
        {
            return Err(fail("unsupported logical shape"));
        }
    }
    if by_name.contains_key(PATCH) && !object(g, "glm_vq.logical_shapes")?.contains_key(PATCH) {
        return Err(fail("missing patch logical shape"));
    }
    Ok(())
}

/// Only the GVQ1 format opts into this validator. Unknown private tensors
/// cannot silently inherit the generic parser's zero-byte fallback.
pub(super) fn validate_part(
    g: &GgufFile,
    tensors: &mut [TensorInfo],
    dir_end: usize,
    data_pos: u64,
) -> Result<()> {
    let requested = g.get_string("general.architecture") == Some(b"glm5_next")
        || g.kv_entries()
            .iter()
            .any(|e| g.key_bytes(e).starts_with(b"glm_vq."))
        || tensors.iter().any(|t| matches!(t.typ, FP8 | CODEBOOK | VQ));
    if !requested {
        return Ok(());
    }
    if g.get_string("general.architecture") != Some(b"glm5_next")
        || g.alignment != ALIGN
        || g.get_u32("glm_vq.format_version") != Some(FORMAT_VERSION)
        || g.split_count() > 1
    {
        return Err(fail("unsupported format/version/alignment"));
    }
    let mut keys = HashSet::new();
    for entry in g.kv_entries() {
        if !keys.insert(g.key_bytes(entry)) {
            return Err(fail("duplicate metadata"));
        }
    }
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
            F16 | BF16 => t.elements.checked_mul(2).ok_or(TensorError::Overflow)?,
            FP8 | CODEBOOK => t.elements,
            VQ => {
                let header = Header::parse(g.as_bytes(), t)?;
                let bytes = header.size;
                headers.push((i, header));
                bytes
            }
            _ => return Err(fail("unsupported tensor type")),
        };
        raw(g.as_bytes(), t)?;
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
