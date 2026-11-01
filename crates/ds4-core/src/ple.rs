//! Qwen SSD-PLE sidecar selection and manifest validation. Native keeps the
//! bounded page cache (O_DIRECT pages, row leases released by CUDA stream
//! callbacks) and rechecks only read bounds (`ds4_ple_store_open_host`).
//! The checks mirror the deprecated C parser in ds4_ple.c, which C hosts
//! still use; a bad sidecar now fails before the weight map instead of
//! after it.
//!
//! ```text
//! DS4_QWEN_PLE_DIR set:  <dir>/ple-manifest.json
//! otherwise:             <gguf dir>/ple/ple-manifest.json
//! payload root:          V1 = that dir, V2 = the manifest's dir
//! ```

use std::ffi::CString;
use std::ops::RangeInclusive;
use std::path::{Component, Path, PathBuf};

use ds4_sys::{ds4_host_ple, ds4_host_ple_file, ds4_host_ple_part};
use serde::Deserialize;

use crate::Error;

const DIR_ENV: &str = "DS4_QWEN_PLE_DIR";
const CACHE_ENV: &str = "DS4_QWEN_PLE_CACHE_MB";
const WORKERS_ENV: &str = "DS4_QWEN_PLE_WORKERS";
const SELECTED_MANIFEST: &str = "ple-manifest.json";
const DEFAULT_MANIFEST: &str = "ple/ple-manifest.json";

const DEFAULT_CACHE_MB: u32 = 2048;
const CACHE_MB_CHOICES: [u32; 3] = [512, 1024, 2048];
const DEFAULT_WORKERS: u32 = 32;
const WORKER_RANGE: RangeInclusive<u32> = 1..=64;

const PAGE: u64 = 4096;
const MANIFEST_MAX_BYTES: u64 = 16 << 20;
const PATH_CAP: usize = 256;
const ROW_DIM: u64 = 160;
const BF16_STRIDE: u32 = 320;
const FP8_STRIDE: u32 = 160;
const NGRAM: u64 = 3;
const HEADS_PER_NGRAM: u64 = 8;
const HEADS: usize = 16;
const PARTS: usize = 128;
const FILES: usize = 4;
const USABLE_ROWS: u64 = 320_001_446;
const PADDED_ROWS: u64 = 320_001_536;
const BF16_FILE_BYTES: u64 = 102_400_786_432;
const FP8_FILE_BYTES: u64 = 51_200_393_216;
const VOCAB: u32 = 248_320;
const EOS: u32 = 248_044;
const HASH_IMPLEMENTATION_SHA256: &str =
    "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459";
const STORAGE: &str = "ssd_backed_bounded_page_cache";
const PATHS_RELATIVE_TO: &str = "directory containing ple-manifest.json";

/// V2 is the published official sidecar, not a generic FP8 table:
/// self-consistent but different controls would silently change rows.
const FP8_MULTIPLIERS: [u64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];
const FP8_HEAD_VOCAB: [u64; HEADS] = [
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
    20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
];
const FP8_ROWS_PER_PART: u64 = 2_500_012;
const FP8_PARTS_PER_FILE: usize = 32;
const FP8_SCALE_BITS: u16 = 0x3951;

/// Borrowed `&str` fields reject escaped strings, as the C parser does;
/// serde rejects duplicate keys and skips unknown ones.
#[derive(Deserialize)]
struct Manifest<'a> {
    format_version: u32,
    artifact_variant: &'a str,
    byte_order: &'a str,
    storage: &'a str,
    storage_dtype: &'a str,
    alignment_bytes: u64,
    row_stride_bytes: u32,
    embedding_row_dimension: u64,
    ngram_size: u64,
    heads_per_ngram: u64,
    number_of_ngram_heads: u64,
    number_of_ple_layers: u64,
    logical_shard_count: u64,
    physical_file_count: u64,
    usable_vocabulary_rows: u64,
    padded_vocabulary_rows: u64,
    total_payload_bytes: u64,
    total_file_bytes_including_alignment: u64,
    layer_multipliers: [u64; 3],
    per_head_vocabulary_sizes: [u64; HEADS],
    per_head_offsets: [u64; HEADS],
    #[serde(borrow)]
    logical_parts: Vec<Part<'a>>,
    #[serde(borrow)]
    physical_files: Vec<File<'a>>,
    #[serde(borrow)]
    hash_reference: HashReference<'a>,
    #[serde(borrow)]
    quantization: Option<Quantization<'a>>,
    physical_paths_relative_to: Option<&'a str>,
}

#[derive(Deserialize)]
struct Part<'a> {
    logical_part: u32,
    physical_file_index: u32,
    physical_file: &'a str,
    global_row_start: u64,
    rows: u64,
    file_offset: u64,
    payload_bytes: u64,
    row_stride_bytes: u32,
    embedding_row_dimension: u64,
}

#[derive(Deserialize)]
struct File<'a> {
    index: u32,
    path: &'a str,
    file_bytes: u64,
    payload_bytes: u64,
}

#[derive(Deserialize)]
struct HashReference<'a> {
    implementation_sha256: &'a str,
}

#[derive(Deserialize)]
struct Quantization<'a> {
    format: &'a str,
    scheme: &'a str,
    scale_is_inverse: bool,
    #[serde(borrow)]
    scale: Scale<'a>,
}

#[derive(Deserialize)]
struct Scale<'a> {
    dtype: &'a str,
    path: &'a str,
    file_bytes: u64,
    shape: [u64; 1],
    bits_hex: &'a str,
}

/// A validated sidecar and the cache policy, ready for native open.
pub(crate) struct Sidecar {
    root: CString,
    names: Vec<CString>,
    desc: ds4_host_ple,
}

impl Sidecar {
    /// Borrowed by native for one open; pointers live as long as `self`.
    pub(crate) fn ffi(&mut self) -> *const ds4_host_ple {
        self.desc.root = self.root.as_ptr();
        for (file, name) in self.desc.files.iter_mut().zip(&self.names) {
            file.name = name.as_ptr();
        }
        &self.desc
    }
}

fn fail(message: impl std::fmt::Display) -> Error {
    Error {
        code: 1,
        message: format!("Qwen PLE: {message}"),
    }
}

/// Resolves the sidecar beside `model_path` (or `DS4_QWEN_PLE_DIR`),
/// validates it and reads the cache policy, warning on ignored values.
pub(crate) fn resolve(model_path: &Path) -> Result<Sidecar, Error> {
    let selected = std::env::var_os(DIR_ENV).filter(|dir| !dir.is_empty());
    let (artifact_root, manifest_path) = match selected {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            let manifest = dir.join(SELECTED_MANIFEST);
            (dir, manifest)
        }
        None => {
            let dir = model_path
                .parent()
                .filter(|dir| !dir.as_os_str().is_empty())
                .unwrap_or(Path::new("."))
                .to_path_buf();
            let manifest = dir.join(DEFAULT_MANIFEST);
            (dir, manifest)
        }
    };

    let (cache_mb, cache_warning) = cache_mb();
    let (workers, workers_warning) = env_u32(WORKERS_ENV, DEFAULT_WORKERS, WORKER_RANGE);
    for warning in [cache_warning, workers_warning].into_iter().flatten() {
        eprintln!("{warning}");
    }

    let text = read_manifest(&manifest_path)?;
    let manifest: Manifest = serde_json::from_slice(&text)
        .map_err(|e| fail(format!("{}: {e}", manifest_path.display())))?;
    check(&manifest).map_err(fail)?;

    // V1 payload paths belong to the artifact root; V2 paths to the
    // manifest directory, so one FP8 sidecar serves several models.
    let root = match manifest.format_version {
        2 => manifest_path
            .parent()
            .unwrap_or(Path::new("/"))
            .to_path_buf(),
        _ => artifact_root,
    };
    let scale_bits = match &manifest.quantization {
        Some(quant) => read_scale(&root, &quant.scale)?,
        None => 0,
    };
    for file in &manifest.physical_files {
        let path = root.join(file.path);
        let len = std::fs::metadata(&path)
            .map_err(|e| fail(format!("{}: {e}", path.display())))?
            .len();
        if len != file.file_bytes {
            return Err(fail(format!(
                "{} has {len} bytes, manifest says {}",
                path.display(),
                file.file_bytes
            )));
        }
    }

    let mut desc = ds4_host_ple {
        root: std::ptr::null(),
        version: manifest.format_version,
        row_stride: manifest.row_stride_bytes,
        usable_rows: manifest.usable_vocabulary_rows,
        padded_rows: manifest.padded_vocabulary_rows,
        vocab: VOCAB,
        eos: EOS,
        multipliers: manifest.layer_multipliers,
        head_vocab: manifest.per_head_vocabulary_sizes,
        head_offset: manifest.per_head_offsets,
        scale_bits: u32::from(scale_bits),
        workers,
        cache_bytes: u64::from(cache_mb) << 20,
        parts: [ds4_host_ple_part::default(); PARTS],
        files: [ds4_host_ple_file {
            name: std::ptr::null(),
            bytes: 0,
        }; FILES],
    };
    for (dst, part) in desc.parts.iter_mut().zip(&manifest.logical_parts) {
        *dst = ds4_host_ple_part {
            row_start: part.global_row_start,
            rows: part.rows,
            offset: part.file_offset,
            file: part.physical_file_index,
            reserved: 0,
        };
    }
    let mut names = Vec::with_capacity(FILES);
    for (dst, file) in desc.files.iter_mut().zip(&manifest.physical_files) {
        dst.bytes = file.file_bytes;
        names.push(CString::new(file.path).map_err(fail)?);
    }
    let root = CString::new(root.as_os_str().as_encoded_bytes()).map_err(fail)?;
    Ok(Sidecar { root, names, desc })
}

/// `DS4_QWEN_PLE_CACHE_MB` as C hosts read it: 512, 1024 or 2048 MiB, else
/// 2048 with the warning to print.
pub(crate) fn cache_mb() -> (u32, Option<String>) {
    let (mb, warning) = env_u32(CACHE_ENV, DEFAULT_CACHE_MB, 512..=DEFAULT_CACHE_MB);
    if warning.is_some() || CACHE_MB_CHOICES.contains(&mb) {
        return (mb, warning);
    }
    let warning =
        format!("ds4: ignoring {CACHE_ENV}={mb} (supported: 512, 1024, 2048); using 2048");
    (DEFAULT_CACHE_MB, Some(warning))
}

/// `strtoul`-compatible: leading blanks and `+` pass, anything else falls back.
fn env_u32(name: &str, default: u32, range: RangeInclusive<u32>) -> (u32, Option<String>) {
    let Some(raw) = std::env::var_os(name).filter(|v| !v.is_empty()) else {
        return (default, None);
    };
    let raw = raw.to_string_lossy();
    match raw.trim_start().parse::<u32>() {
        Ok(value) if range.contains(&value) => (value, None),
        _ => {
            let (lo, hi) = (range.start(), range.end());
            (
                default,
                Some(format!(
                    "ds4: ignoring invalid {name}='{raw}' (expected {lo}..{hi})"
                )),
            )
        }
    }
}

fn read_manifest(path: &Path) -> Result<Vec<u8>, Error> {
    let len = std::fs::metadata(path)
        .map_err(|e| fail(format!("{}: {e}", path.display())))?
        .len();
    if len == 0 || len > MANIFEST_MAX_BYTES {
        return Err(fail(format!("{} must hold 1..16 MiB", path.display())));
    }
    std::fs::read(path).map_err(|e| fail(format!("{}: {e}", path.display())))
}

/// V2: the 2-byte scale file must equal the official bits.
fn read_scale(root: &Path, scale: &Scale) -> Result<u16, Error> {
    let bits = scale
        .bits_hex
        .strip_prefix("0x")
        .filter(|hex| hex.len() == 4)
        .and_then(|hex| u16::from_str_radix(hex, 16).ok());
    if bits != Some(FP8_SCALE_BITS) || !safe_relative(scale.path) {
        return Err(fail(
            "FP8 scale must match official BF16 bits 0x3951 with a relative path",
        ));
    }
    let path = root.join(scale.path);
    let data = std::fs::read(&path).map_err(|e| fail(format!("{}: {e}", path.display())))?;
    if data != FP8_SCALE_BITS.to_le_bytes() {
        return Err(fail("FP8 scale file does not match the manifest"));
    }
    Ok(FP8_SCALE_BITS)
}

fn safe_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && path.len() < PATH_CAP
        && path
            .split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
        && Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

fn require(ok: bool, what: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(what.to_string())
    }
}

/// Every check of the C parser (ds4_ple.c manifest_parse and
/// manifest_validate_layout) except the JSON grammar, which serde owns.
fn check(m: &Manifest) -> Result<(), String> {
    let fp8 = m.format_version == 2;
    require(
        m.format_version == 1 || fp8,
        "format_version must be 1 or 2",
    )?;
    require(m.byte_order == "little", "byte_order must be little")?;
    require(m.storage == STORAGE, "unsupported storage")?;
    require(m.alignment_bytes == PAGE, "alignment_bytes must be 4096")?;
    require(
        m.embedding_row_dimension == ROW_DIM,
        "embedding_row_dimension must be 160",
    )?;
    require(
        m.ngram_size == NGRAM
            && m.heads_per_ngram == HEADS_PER_NGRAM
            && m.number_of_ngram_heads == HEADS as u64
            && m.number_of_ple_layers == 1
            && m.logical_shard_count == PARTS as u64
            && m.physical_file_count == FILES as u64,
        "unsupported n-gram topology",
    )?;
    require(
        m.hash_reference.implementation_sha256 == HASH_IMPLEMENTATION_SHA256,
        "hash implementation differs from the reference",
    )?;
    require(
        m.quantization.is_some() == fp8 && m.physical_paths_relative_to.is_some() == fp8,
        "missing required top-level fields",
    )?;
    if let Some(relative) = m.physical_paths_relative_to {
        require(
            relative == PATHS_RELATIVE_TO,
            "physical_paths_relative_to is unsupported",
        )?;
    }
    let format_ok = if fp8 {
        m.artifact_variant == "PLE-FP8"
            && m.storage_dtype == "FP8_E4M3FN"
            && m.row_stride_bytes == FP8_STRIDE
    } else {
        matches!(
            m.artifact_variant,
            "MQ-Q5-SSD-PLE-BF16" | "MQ-Q6-SSD-PLE-BF16"
        ) && m.storage_dtype == "BF16"
            && m.row_stride_bytes == BF16_STRIDE
    };
    require(format_ok, "variant, dtype and row stride disagree")?;
    if let Some(quant) = &m.quantization {
        let scale = &quant.scale;
        require(
            quant.format == "E4M3FN"
                && quant.scheme == "per_tensor"
                && !quant.scale_is_inverse
                && scale.dtype == "BF16"
                && scale.file_bytes == 2
                && scale.shape == [1],
            "unsupported FP8 quantization",
        )?;
    }
    check_hash(m)?;
    check_layout(m, fp8)
}

/// ds4_ple_hash_config_validate on the fixed Qwen vocabulary.
fn check_hash(m: &Manifest) -> Result<(), String> {
    let max_token = u64::from(VOCAB - 1);
    for &multiplier in &m.layer_multipliers {
        require(multiplier & 1 == 1, "multipliers must be positive and odd")?;
        require(
            multiplier <= i64::MAX as u64 / max_token,
            "multiplier can overflow int64",
        )?;
    }
    let mut next = 0u64;
    for (&size, &offset) in m.per_head_vocabulary_sizes.iter().zip(&m.per_head_offsets) {
        require(
            size != 0 && offset == next,
            "head vocabularies are not contiguous",
        )?;
        next = next
            .checked_add(size)
            .ok_or("head vocabulary sum overflows")?;
    }
    require(
        next == m.usable_vocabulary_rows,
        "head vocabularies do not match usable rows",
    )
}

fn check_layout(m: &Manifest, fp8: bool) -> Result<(), String> {
    let stride = u64::from(m.row_stride_bytes);
    let file_bytes = if fp8 { FP8_FILE_BYTES } else { BF16_FILE_BYTES };
    require(
        m.usable_vocabulary_rows == USABLE_ROWS
            && m.padded_vocabulary_rows == PADDED_ROWS
            && m.total_payload_bytes == PADDED_ROWS * stride
            && m.total_file_bytes_including_alignment == file_bytes,
        "aggregate sizes do not match the storage reference",
    )?;
    require(
        m.logical_parts.len() == PARTS && m.physical_files.len() == FILES,
        "manifest must list 128 parts and 4 files",
    )?;

    let (mut rows, mut payload) = (0u64, 0u64);
    for (i, part) in m.logical_parts.iter().enumerate() {
        require(
            part.logical_part as usize == i
                && part.global_row_start == rows
                && part.rows != 0
                && (part.physical_file_index as usize) < FILES
                && part.file_offset % PAGE == 0
                && part.row_stride_bytes == m.row_stride_bytes
                && part.embedding_row_dimension == ROW_DIM
                && part.rows.checked_mul(stride) == Some(part.payload_bytes),
            &format!("logical part {i} is invalid"),
        )?;
        rows = rows.checked_add(part.rows).ok_or("part rows overflow")?;
        payload = payload
            .checked_add(part.payload_bytes)
            .ok_or("part payload overflows")?;
    }
    require(
        rows == m.padded_vocabulary_rows && payload == m.total_payload_bytes,
        "logical parts do not match manifest aggregates",
    )?;

    let (mut bytes, mut file_payload) = (0u64, 0u64);
    for (i, file) in m.physical_files.iter().enumerate() {
        require(
            file.index as usize == i
                && safe_relative(file.path)
                && file.file_bytes != 0
                && file.file_bytes % PAGE == 0,
            &format!("physical file {i} is invalid"),
        )?;
        bytes = bytes
            .checked_add(file.file_bytes)
            .ok_or("file bytes overflow")?;
        file_payload = file_payload
            .checked_add(file.payload_bytes)
            .ok_or("file payload overflows")?;
    }
    require(
        bytes == m.total_file_bytes_including_alignment && file_payload == m.total_payload_bytes,
        "physical files do not match manifest aggregates",
    )?;
    for (i, part) in m.logical_parts.iter().enumerate() {
        let file = &m.physical_files[part.physical_file_index as usize];
        require(
            part.physical_file == file.path
                && part.file_offset <= file.file_bytes
                && part.payload_bytes <= file.file_bytes - part.file_offset,
            &format!("logical part {i} exceeds its physical file"),
        )?;
    }
    if !fp8 {
        return Ok(());
    }

    require(
        m.layer_multipliers == FP8_MULTIPLIERS && m.per_head_vocabulary_sizes == FP8_HEAD_VOCAB,
        "FP8 hash controls differ from the official reference",
    )?;
    let part_payload = FP8_ROWS_PER_PART * u64::from(FP8_STRIDE);
    let aligned = part_payload.next_multiple_of(PAGE);
    for (i, part) in m.logical_parts.iter().enumerate() {
        require(
            part.rows == FP8_ROWS_PER_PART
                && part.physical_file_index as usize == i / FP8_PARTS_PER_FILE
                && part.file_offset == (i % FP8_PARTS_PER_FILE) as u64 * aligned,
            &format!("FP8 part {i} differs from the official layout"),
        )?;
    }
    for (i, file) in m.physical_files.iter().enumerate() {
        require(
            file.file_bytes == FP8_PARTS_PER_FILE as u64 * aligned
                && file.payload_bytes == FP8_PARTS_PER_FILE as u64 * part_payload,
            &format!("FP8 file {i} differs from the official layout"),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A V2 manifest in the official layout, as `serde_json::Value`.
    fn fp8_manifest() -> serde_json::Value {
        let payload = FP8_ROWS_PER_PART * u64::from(FP8_STRIDE);
        let aligned = payload.next_multiple_of(PAGE);
        let name = |f: usize| format!("ple-fp8-0000{}-of-00004.bin", f + 1);
        let parts: Vec<_> = (0..PARTS)
            .map(|i| {
                serde_json::json!({
                    "logical_part": i, "physical_file_index": i / 32, "physical_file": name(i / 32),
                    "global_row_start": i as u64 * FP8_ROWS_PER_PART, "rows": FP8_ROWS_PER_PART,
                    "file_offset": (i % 32) as u64 * aligned, "payload_bytes": payload,
                    "row_stride_bytes": FP8_STRIDE, "embedding_row_dimension": 160, "sha256": "x"
                })
            })
            .collect();
        let files: Vec<_> = (0..FILES)
            .map(|i| serde_json::json!({"index": i, "path": name(i), "file_bytes": 32 * aligned, "payload_bytes": 32 * payload}))
            .collect();
        let mut offsets = Vec::new();
        let mut next = 0;
        for size in FP8_HEAD_VOCAB {
            offsets.push(next);
            next += size;
        }
        serde_json::json!({
            "format_version": 2, "artifact_variant": "PLE-FP8", "byte_order": "little",
            "storage": STORAGE, "storage_dtype": "FP8_E4M3FN", "alignment_bytes": 4096,
            "row_stride_bytes": 160, "embedding_row_dimension": 160, "ngram_size": 3,
            "heads_per_ngram": 8, "number_of_ngram_heads": 16, "number_of_ple_layers": 1,
            "logical_shard_count": 128, "physical_file_count": 4,
            "usable_vocabulary_rows": USABLE_ROWS, "padded_vocabulary_rows": PADDED_ROWS,
            "total_payload_bytes": PADDED_ROWS * 160, "total_file_bytes_including_alignment": FP8_FILE_BYTES,
            "layer_multipliers": FP8_MULTIPLIERS, "per_head_vocabulary_sizes": FP8_HEAD_VOCAB,
            "per_head_offsets": offsets, "logical_parts": parts, "physical_files": files,
            "hash_reference": {"implementation_sha256": HASH_IMPLEMENTATION_SHA256, "implementation": "x"},
            "quantization": {"format": "E4M3FN", "scheme": "per_tensor", "scale_is_inverse": false,
                "scale": {"dtype": "BF16", "path": "ple-fp8-weight-scale.bf16.bin", "file_bytes": 2,
                          "shape": [1], "bits_hex": "0x3951"}},
            "physical_paths_relative_to": PATHS_RELATIVE_TO, "unknown_key": {"nested": [1, 2]}
        })
    }

    fn checked(value: &serde_json::Value) -> Result<(), String> {
        let text = serde_json::to_vec(value).unwrap();
        let manifest: Manifest = serde_json::from_slice(&text).map_err(|e| e.to_string())?;
        check(&manifest)
    }

    #[test]
    fn official_fp8_layout_passes() {
        checked(&fp8_manifest()).unwrap();
    }

    #[test]
    fn mutations_fail() {
        type Mutate = fn(&mut serde_json::Value);
        let cases: [(&str, Mutate); 10] = [
            ("version", |m| m["format_version"] = 3.into()),
            ("v1 with quantization", |m| m["format_version"] = 1.into()),
            ("missing paths key", |m| {
                m.as_object_mut()
                    .unwrap()
                    .remove("physical_paths_relative_to");
            }),
            ("stride", |m| m["row_stride_bytes"] = 320.into()),
            ("hash sha", |m| {
                m["hash_reference"]["implementation_sha256"] = "00".into()
            }),
            ("multiplier", |m| {
                m["layer_multipliers"][0] = 23_703_573_157_771u64.into()
            }),
            ("part offset", |m| {
                m["logical_parts"][5]["file_offset"] = 4096.into()
            }),
            ("part file name", |m| {
                m["logical_parts"][0]["physical_file"] = "other.bin".into()
            }),
            ("unsafe path", |m| {
                m["physical_files"][0]["path"] = "../x.bin".into()
            }),
            ("scale inverse", |m| {
                m["quantization"]["scale_is_inverse"] = true.into()
            }),
        ];
        for (name, mutate) in cases {
            let mut m = fp8_manifest();
            mutate(&mut m);
            assert!(checked(&m).is_err(), "{name} passed");
        }
    }

    #[test]
    fn grammar_matches_c() {
        let text = serde_json::to_string(&fp8_manifest()).unwrap();
        // Required strings may not carry escapes.
        let escaped = text.replace("\"PLE-FP8\"", "\"PLE\\u002dFP8\"");
        assert!(serde_json::from_str::<Manifest>(&escaped).is_err());
        // Duplicate required keys are rejected.
        let duplicate = text.replacen("{", "{\"format_version\":2,", 1);
        assert!(serde_json::from_str::<Manifest>(&duplicate).is_err());
        // Integers only.
        let float = text.replace("\"alignment_bytes\":4096", "\"alignment_bytes\":4096.0");
        assert!(serde_json::from_str::<Manifest>(&float).is_err());
    }

    #[test]
    fn env_values_parse_like_strtoul() {
        assert_eq!(env_u32("DS4_PLE_TEST_UNSET_VAR", 32, 1..=64), (32, None));
        assert!(safe_relative("ple/ple-bf16-00001-of-00004.bin"));
        assert!(!safe_relative("./x") && !safe_relative("a//b") && !safe_relative("/abs"));
    }

    /// The published sidecar on this host, when present.
    #[test]
    #[ignore]
    fn real_fp8_manifest() {
        let dir = std::env::var("DS4_TEST_PLE_DIR").expect("DS4_TEST_PLE_DIR");
        let text = read_manifest(&Path::new(&dir).join(SELECTED_MANIFEST)).unwrap();
        let manifest: Manifest = serde_json::from_slice(&text).unwrap();
        check(&manifest).unwrap();
        let start = std::time::Instant::now();
        for _ in 0..50 {
            let manifest: Manifest = serde_json::from_slice(&text).unwrap();
            check(&manifest).unwrap();
        }
        eprintln!("parse+check: {:?} per manifest", start.elapsed() / 50);
    }
}
