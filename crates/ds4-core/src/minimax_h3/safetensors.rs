//! safetensors headers and shard directories for MiniMax-H3.
//!
//! Byte contract and validation mirror the reference engine at
//! `/data/h3.c` `eae5d54` (`h3_safetensors.c`, `h3_weights.c`); the line
//! references live in the parent module's header. Deviations are named where
//! they occur.
//!
//! ```text
//! file    = u64le header_len | header_len bytes of JSON | data area
//! header  = { "__metadata__": {...}?, "<tensor>": { "dtype": str,
//!             "shape": [u64...], "data_offsets": [begin, end] }, ... }
//! offset  = 8 + header_len + begin          (the reference's file_offset)
//! ```
//!
//! The JSON parse uses `serde_json` instead of the reference's hand parser.
//! The accepted grammar is the same for every header that parser accepts;
//! serde additionally accepts duplicate keys and numbers written in exponent
//! form, which the reference rejects. Neither changes a valid header's
//! interpretation, and every structural rule below is checked explicitly.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// `H3_ST_MAX_HEADER` (`h3_safetensors.c:14`).
pub const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
/// The little-endian length prefix before the JSON header.
pub const LENGTH_BYTES: u64 = 8;
/// `h3_parse_shape` refuses a ninth dimension (`h3_safetensors.c:252`).
pub const MAX_RANK: usize = 8;

/// `h3_dtype` (`h3_safetensors.h:9-23`), minus the UNKNOWN failure case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    I8,
    U8,
    I16,
    U16,
    F16,
    Bf16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
}

impl Dtype {
    /// `h3_parse_dtype` (`h3_safetensors.c:182-197`).
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "BOOL" => Self::Bool,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "F32" => Self::F32,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "F64" => Self::F64,
            _ => return None,
        })
    }

    /// `h3_dtype_name` (`h3_safetensors.c:220-238`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::F32 => "F32",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::F64 => "F64",
        }
    }

    /// `h3_dtype_size` (`h3_safetensors.c:200-217`).
    pub fn size(self) -> u64 {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// `h3_st_tensor` (`h3_safetensors.h:26-34`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// `data_offsets[0]`, relative to the data area.
    pub data_begin: u64,
    /// `data_offsets[1]`, relative to the data area.
    pub data_end: u64,
    /// `8 + header_size + data_begin`, the absolute read offset.
    pub file_offset: u64,
}

impl Tensor {
    /// `h3_st_tensor_elements` (`h3_safetensors.c:480-489`).
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The `data_offsets` span, which `read_shard` proved equal to
    /// `elements * dtype.size()`.
    pub fn bytes(&self) -> u64 {
        self.data_end - self.data_begin
    }
}

/// `h3_st_header` (`h3_safetensors.h:36-42`).
#[derive(Debug)]
pub struct Shard {
    pub path: PathBuf,
    pub file_size: u64,
    pub header_size: u64,
    pub tensors: Vec<Tensor>,
}

impl Shard {
    /// `h3_st_find` (`h3_safetensors.c:472-478`).
    pub fn find(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// `h3_st_read_data` (`h3_safetensors.c:492-534`): the buffer must be
    /// exactly the tensor's byte span.
    pub fn read_data(&self, tensor: &Tensor, buffer: &mut [u8]) -> Result<(), StError> {
        let expected = tensor.bytes();
        let got = buffer.len() as u64;

        if got != expected {
            return Err(StError::SizeMismatch {
                name: tensor.name.clone(),
                expected,
                got,
            });
        }

        let file = File::open(&self.path).map_err(|error| StError::Io {
            path: self.path.clone(),
            detail: error.to_string(),
        })?;

        file.read_exact_at(buffer, tensor.file_offset)
            .map_err(|error| StError::Io {
                path: self.path.clone(),
                detail: error.to_string(),
            })
    }

    /// Total tensor bytes in this shard (`inventory_dir`'s `tensor_bytes`).
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(Tensor::bytes).sum()
    }
}

/// `h3_component_info` (`h3.h`), the directory totals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Inventory {
    pub files: u64,
    pub bytes: u64,
    pub tensors: u64,
    pub tensor_bytes: u64,
}

/// A sorted multi-shard directory (`h3_weight_store`, `h3_weights.c:44-114`).
#[derive(Debug)]
pub struct Store {
    shards: Vec<Shard>,
}

impl Store {
    pub fn shards(&self) -> &[Shard] {
        &self.shards
    }

    /// `h3_weight_find` (`h3_weights.c:129-142`): first shard in sorted path
    /// order wins, and the shard travels back with the tensor.
    pub fn find(&self, name: &str) -> Option<(&Shard, &Tensor)> {
        self.shards
            .iter()
            .find_map(|shard| shard.find(name).map(|tensor| (shard, tensor)))
    }

    pub fn tensor_count(&self) -> u64 {
        self.shards
            .iter()
            .map(|shard| shard.tensors.len() as u64)
            .sum()
    }

    pub fn tensor_bytes(&self) -> u64 {
        self.shards.iter().map(Shard::tensor_bytes).sum()
    }
}

#[derive(Debug)]
pub enum StError {
    Io {
        path: PathBuf,
        detail: String,
    },
    /// A file shorter than the 8-byte length prefix.
    Truncated(PathBuf),
    /// `header_size > H3_ST_MAX_HEADER` or beyond the file.
    HeaderSize {
        path: PathBuf,
        header_size: u64,
        file_size: u64,
    },
    /// The header is not a JSON object, or a value has the wrong JSON type.
    Header {
        path: PathBuf,
        detail: String,
    },
    /// `h3_parse_dtype` returned UNKNOWN.
    Dtype {
        name: String,
        dtype: String,
    },
    Rank {
        name: String,
        rank: usize,
    },
    /// `data_offsets` is not exactly two unsigned integers.
    Offsets {
        name: String,
        detail: String,
    },
    /// `data_offsets` falls outside the data area.
    Outside {
        name: String,
        begin: u64,
        end: u64,
        available: u64,
    },
    Overflow {
        name: String,
    },
    /// `elements * dtype_size != data_end - data_begin`.
    ShapeBytes {
        name: String,
        expected: u64,
        span: u64,
    },
    SizeMismatch {
        name: String,
        expected: u64,
        got: u64,
    },
    NoShards(PathBuf),
    Index {
        path: PathBuf,
        detail: String,
    },
    /// The index names a tensor its shard does not hold.
    MissingTensor {
        name: String,
        shard: String,
    },
    /// `weight_map[name]` names a different shard than the one holding it.
    WrongShard {
        name: String,
        indexed: String,
        actual: String,
    },
    /// A header tensor absent from `weight_map`.
    Unindexed(String),
    /// `metadata.total_size` disagrees with the header byte total.
    TotalSize {
        expected: u64,
        got: u64,
    },
}

impl StError {
    /// Short machine-readable token, matching this crate's error style.
    pub fn token(&self) -> String {
        match self {
            Self::Io { path, detail } => format!("st-io {} {detail}", path.display()),
            Self::Truncated(path) => format!("st-truncated {}", path.display()),
            Self::HeaderSize { path, .. } => format!("st-header-size {}", path.display()),
            Self::Header { path, detail } => format!("st-header {} {detail}", path.display()),
            Self::Dtype { name, dtype } => format!("st-dtype {name} {dtype}"),
            Self::Rank { name, rank } => format!("st-rank {name} {rank}"),
            Self::Offsets { name, detail } => format!("st-offsets {name} {detail}"),
            Self::Outside { name, .. } => format!("st-outside {name}"),
            Self::Overflow { name } => format!("st-overflow {name}"),
            Self::ShapeBytes { name, .. } => format!("st-shape-bytes {name}"),
            Self::SizeMismatch { name, .. } => format!("st-size {name}"),
            Self::NoShards(path) => format!("st-no-shards {}", path.display()),
            Self::Index { path, detail } => format!("st-index {} {detail}", path.display()),
            Self::MissingTensor { name, .. } => format!("st-missing-tensor {name}"),
            Self::WrongShard { name, .. } => format!("st-wrong-shard {name}"),
            Self::Unindexed(name) => format!("st-unindexed {name}"),
            Self::TotalSize { .. } => "st-total-size".into(),
        }
    }
}

impl fmt::Display for StError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, detail } => write!(formatter, "{}: {detail}", path.display()),
            Self::Truncated(path) => write!(formatter, "{}: shorter than 8 bytes", path.display()),
            Self::HeaderSize {
                path,
                header_size,
                file_size,
            } => write!(
                formatter,
                "{}: header size {header_size} exceeds {file_size}",
                path.display()
            ),
            Self::Header { path, detail } => write!(formatter, "{}: {detail}", path.display()),
            Self::Dtype { name, dtype } => {
                write!(formatter, "{name}: unknown dtype {dtype}")
            }
            Self::Rank { name, rank } => {
                write!(formatter, "{name}: rank {rank} exceeds {MAX_RANK}")
            }
            Self::Offsets { name, detail } => write!(formatter, "{name}: {detail}"),
            Self::Outside {
                name,
                begin,
                end,
                available,
            } => write!(
                formatter,
                "{name}: offsets {begin}..{end} outside a {available}-byte data area"
            ),
            Self::Overflow { name } => write!(formatter, "{name}: element count overflows"),
            Self::ShapeBytes {
                name,
                expected,
                span,
            } => write!(
                formatter,
                "{name}: shape needs {expected} bytes, offsets span {span}"
            ),
            Self::SizeMismatch {
                name,
                expected,
                got,
            } => write!(formatter, "{name}: expected {expected} bytes, got {got}"),
            Self::NoShards(path) => write!(formatter, "{}: no safetensors files", path.display()),
            Self::Index { path, detail } => write!(formatter, "{}: {detail}", path.display()),
            Self::MissingTensor { name, shard } => {
                write!(formatter, "{name}: absent from {shard}")
            }
            Self::WrongShard {
                name,
                indexed,
                actual,
            } => write!(formatter, "{name}: indexed in {indexed}, found in {actual}"),
            Self::Unindexed(name) => write!(formatter, "{name}: absent from weight_map"),
            Self::TotalSize { expected, got } => {
                write!(
                    formatter,
                    "total_size {expected} against {got} header bytes"
                )
            }
        }
    }
}

impl std::error::Error for StError {}

/// `h3_st_read_header` (`h3_safetensors.c:367-460`).
///
/// Tensors come back sorted by name so two inventories compare textually;
/// the reference keeps them in header order, which a diff sorts anyway.
pub fn read_shard(path: &Path) -> Result<Shard, StError> {
    let file = File::open(path).map_err(|error| StError::Io {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })?;
    let file_size = file
        .metadata()
        .map_err(|error| StError::Io {
            path: path.to_path_buf(),
            detail: error.to_string(),
        })?
        .len();

    if file_size < LENGTH_BYTES {
        return Err(StError::Truncated(path.to_path_buf()));
    }

    let mut prefix = [0u8; 8];
    read_at(&file, &mut prefix, 0, path)?;
    let header_size = u64::from_le_bytes(prefix);

    if header_size > MAX_HEADER_BYTES || header_size > file_size - LENGTH_BYTES {
        return Err(StError::HeaderSize {
            path: path.to_path_buf(),
            header_size,
            file_size,
        });
    }

    let mut json = vec![0u8; header_size as usize];
    read_at(&file, &mut json, LENGTH_BYTES, path)?;

    let value: Value = serde_json::from_slice(&json).map_err(|error| StError::Header {
        path: path.to_path_buf(),
        detail: format!("malformed safetensors JSON header: {error}"),
    })?;

    let object = value.as_object().ok_or_else(|| StError::Header {
        path: path.to_path_buf(),
        detail: "header is not a JSON object".into(),
    })?;

    let data_start = LENGTH_BYTES + header_size;
    let mut tensors = Vec::with_capacity(object.len());

    for (name, entry) in object {
        if name == "__metadata__" {
            continue;
        }
        tensors.push(parse_tensor(name, entry, data_start, file_size).map_err(
            |error| match error {
                StError::Header { detail, .. } => StError::Header {
                    path: path.to_path_buf(),
                    detail: format!("{name}: {detail}"),
                },
                other => other,
            },
        )?);
    }

    tensors.sort_by(|left, right| left.name.cmp(&right.name));

    Ok(Shard {
        path: path.to_path_buf(),
        file_size,
        header_size,
        tensors,
    })
}

/// `h3_parse_tensor` plus `h3_validate_tensor`
/// (`h3_safetensors.c:277-345`).
fn parse_tensor(
    name: &str,
    value: &Value,
    data_start: u64,
    file_size: u64,
) -> Result<Tensor, StError> {
    let entry = value.as_object().ok_or_else(|| StError::Header {
        path: PathBuf::new(),
        detail: "tensor entry is not an object".into(),
    })?;

    let dtype_name = entry
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| incomplete("dtype"))?;
    let dtype = Dtype::parse(dtype_name).ok_or_else(|| StError::Dtype {
        name: name.to_string(),
        dtype: dtype_name.to_string(),
    })?;

    let shape_values = entry
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| incomplete("shape"))?;

    if shape_values.len() > MAX_RANK {
        return Err(StError::Rank {
            name: name.to_string(),
            rank: shape_values.len(),
        });
    }

    let mut shape = Vec::with_capacity(shape_values.len());
    for dimension in shape_values {
        let value = dimension.as_u64().ok_or_else(|| StError::Header {
            path: PathBuf::new(),
            detail: "shape dimension is not an unsigned integer".into(),
        })?;
        shape.push(value);
    }

    let offsets = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| incomplete("data_offsets"))?;

    if offsets.len() != 2 {
        return Err(StError::Offsets {
            name: name.to_string(),
            detail: format!("expected 2 data offsets, got {}", offsets.len()),
        });
    }

    let begin = offsets[0].as_u64().ok_or_else(|| StError::Offsets {
        name: name.to_string(),
        detail: "data_offsets[0] is not an unsigned integer".into(),
    })?;
    let end = offsets[1].as_u64().ok_or_else(|| StError::Offsets {
        name: name.to_string(),
        detail: "data_offsets[1] is not an unsigned integer".into(),
    })?;

    let available = file_size - data_start;

    if end < begin || end > available {
        return Err(StError::Outside {
            name: name.to_string(),
            begin,
            end,
            available,
        });
    }

    let elements = shape
        .iter()
        .try_fold(1u64, |count, dimension| count.checked_mul(*dimension));

    let elements = elements.ok_or_else(|| StError::Overflow {
        name: name.to_string(),
    })?;

    let expected = elements
        .checked_mul(dtype.size())
        .ok_or_else(|| StError::Overflow {
            name: name.to_string(),
        })?;

    if expected != end - begin {
        return Err(StError::ShapeBytes {
            name: name.to_string(),
            expected,
            span: end - begin,
        });
    }

    Ok(Tensor {
        name: name.to_string(),
        dtype,
        shape,
        data_begin: begin,
        data_end: end,
        file_offset: data_start + begin,
    })
}

fn incomplete(field: &str) -> StError {
    StError::Header {
        path: PathBuf::new(),
        detail: format!("incomplete tensor descriptor: {field} is missing"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ds4-h3-st-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn shard_bytes(entries: &[(&str, &str, &[u64], &[u8])]) -> Vec<u8> {
        let mut body = Vec::new();
        let mut fields = Vec::new();

        for (name, dtype, shape, data) in entries {
            let begin = body.len() as u64;
            body.extend_from_slice(data);
            let end = body.len() as u64;
            let shape = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            fields.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape}],\"data_offsets\":[{begin},{end}]}}"
            ));
        }

        let json = format!("{{{}}}", fields.join(","));
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(&body);
        bytes
    }

    #[test]
    fn reads_a_header_and_its_data() {
        let dir = scratch("roundtrip");
        let data: Vec<u8> = (0..64).collect();
        let path = dir.join("a.safetensors");
        fs::write(
            &path,
            shard_bytes(&[("w", "BF16", &[4, 8], &data), ("b", "F32", &[2], &[0; 8])]),
        )
        .unwrap();

        let shard = read_shard(&path).unwrap();
        assert_eq!(shard.tensors.len(), 2);
        assert_eq!(shard.file_size, fs::metadata(&path).unwrap().len());

        let tensor = shard.find("w").unwrap();
        assert_eq!(tensor.dtype, Dtype::Bf16);
        assert_eq!(tensor.shape, vec![4, 8]);
        assert_eq!(tensor.elements(), 32);
        assert_eq!(tensor.bytes(), 64);
        assert_eq!(tensor.file_offset, LENGTH_BYTES + shard.header_size);

        let mut buffer = vec![0u8; 64];
        shard.read_data(tensor, &mut buffer).unwrap();
        assert_eq!(buffer, data);

        assert!(shard.find("absent").is_none());
    }

    #[test]
    fn rejects_a_truncated_file_and_a_giant_header() {
        let dir = scratch("header-size");
        let short = dir.join("short.safetensors");
        fs::write(&short, [0u8; 4]).unwrap();
        assert!(matches!(read_shard(&short), Err(StError::Truncated(_))));

        let giant = dir.join("giant.safetensors");
        let mut bytes = (MAX_HEADER_BYTES + 1).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        fs::write(&giant, bytes).unwrap();
        assert!(matches!(
            read_shard(&giant),
            Err(StError::HeaderSize { .. })
        ));
    }

    #[test]
    fn rejects_offsets_beyond_the_data_area() {
        let dir = scratch("outside");
        let path = dir.join("a.safetensors");
        let json = r#"{"w":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,64]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(&[0u8; 32]);
        fs::write(&path, bytes).unwrap();

        assert!(matches!(read_shard(&path), Err(StError::Outside { .. })));
    }

    #[test]
    fn rejects_a_shape_that_disagrees_with_its_bytes() {
        let dir = scratch("shape-bytes");
        let path = dir.join("a.safetensors");
        let json = r#"{"w":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,32]}}"#;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(&[0u8; 32]);
        fs::write(&path, bytes).unwrap();

        assert!(matches!(read_shard(&path), Err(StError::ShapeBytes { .. })));
    }

    #[test]
    fn rejects_an_unknown_dtype_and_an_overlong_rank() {
        let dir = scratch("descriptor");

        let dtype = dir.join("dtype.safetensors");
        fs::write(&dtype, shard_bytes(&[("w", "F8", &[1], &[0u8; 1])])).unwrap();
        assert!(matches!(read_shard(&dtype), Err(StError::Dtype { .. })));

        let rank = dir.join("rank.safetensors");
        let shape = [1u64; 9];
        fs::write(&rank, shard_bytes(&[("w", "BF16", &shape, &[0u8; 2])])).unwrap();
        assert!(matches!(read_shard(&rank), Err(StError::Rank { .. })));
    }

    #[test]
    fn a_directory_store_is_sorted_and_finds_across_shards() {
        let dir = scratch("store");
        fs::write(
            dir.join("b.safetensors"),
            shard_bytes(&[("second", "BF16", &[1], &[1, 2])]),
        )
        .unwrap();
        fs::write(
            dir.join("a.safetensors"),
            shard_bytes(&[("first", "BF16", &[1], &[3, 4])]),
        )
        .unwrap();
        fs::write(dir.join(".ignored"), b"not a shard").unwrap();

        let store = open_dir(&dir).unwrap();
        let names: Vec<_> = store
            .shards()
            .iter()
            .map(|shard| {
                shard
                    .path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, vec!["a.safetensors", "b.safetensors"]);

        let (shard, tensor) = store.find("second").unwrap();
        assert_eq!(shard.path, dir.join("b.safetensors"));
        assert_eq!(tensor.shape, vec![1]);
        assert_eq!(store.tensor_count(), 2);
        assert_eq!(store.tensor_bytes(), 4);
    }

    #[test]
    fn inventory_counts_files_tensors_and_bytes() {
        let dir = scratch("inventory");
        fs::write(
            dir.join("a.safetensors"),
            shard_bytes(&[("w", "BF16", &[2, 2], &[0u8; 8])]),
        )
        .unwrap();
        fs::write(
            dir.join("b.safetensors"),
            shard_bytes(&[("v", "F32", &[2], &[0u8; 8])]),
        )
        .unwrap();
        fs::write(dir.join(".hidden.safetensors"), b"ignored").unwrap();

        let totals = inventory(&dir).unwrap();
        assert_eq!(totals.files, 2);
        assert_eq!(totals.tensors, 2);
        assert_eq!(totals.tensor_bytes, 16);
        assert_eq!(
            totals.bytes,
            fs::metadata(dir.join("a.safetensors")).unwrap().len()
                + fs::metadata(dir.join("b.safetensors")).unwrap().len()
        );
    }

    #[test]
    fn reconcile_accepts_a_matching_index() {
        let dir = scratch("reconcile-ok");
        fs::write(
            dir.join("a.safetensors"),
            shard_bytes(&[("w", "BF16", &[2], &[0u8; 4])]),
        )
        .unwrap();
        let index = dir.join("model.safetensors.index.json");
        fs::write(
            &index,
            r#"{"metadata":{"total_size":4},"weight_map":{"w":"a.safetensors"}}"#,
        )
        .unwrap();

        let manifest = reconcile(&dir, &index).unwrap();
        assert_eq!(
            manifest,
            Manifest {
                shards: 1,
                tensors: 1,
                tensor_bytes: 4,
            }
        );
    }

    #[test]
    fn reconcile_reports_each_disagreement() {
        let dir = scratch("reconcile-bad");
        fs::write(
            dir.join("a.safetensors"),
            shard_bytes(&[
                ("w", "BF16", &[2], &[0u8; 4]),
                ("x", "BF16", &[1], &[0u8; 2]),
            ]),
        )
        .unwrap();
        fs::write(
            dir.join("b.safetensors"),
            shard_bytes(&[("v", "BF16", &[1], &[0u8; 2])]),
        )
        .unwrap();
        let index = dir.join("model.safetensors.index.json");

        fs::write(
            &index,
            r#"{"weight_map":{"w":"a.safetensors","x":"a.safetensors","v":"b.safetensors"}}"#,
        )
        .unwrap();
        assert!(reconcile(&dir, &index).is_ok());

        fs::write(
            &index,
            r#"{"weight_map":{"w":"a.safetensors","x":"a.safetensors"}}"#,
        )
        .unwrap();
        assert!(matches!(
            reconcile(&dir, &index),
            Err(StError::Unindexed(name)) if name == "v"
        ));

        fs::write(
            &index,
            r#"{"weight_map":{"w":"a.safetensors","x":"a.safetensors","v":"b.safetensors","z":"c.safetensors"}}"#,
        )
        .unwrap();
        assert!(matches!(
            reconcile(&dir, &index),
            Err(StError::Index { .. })
        ));

        fs::write(
            &index,
            r#"{"weight_map":{"w":"a.safetensors","x":"a.safetensors","v":"a.safetensors"}}"#,
        )
        .unwrap();
        assert!(matches!(
            reconcile(&dir, &index),
            Err(StError::WrongShard { .. })
        ));

        fs::write(
            &index,
            r#"{"metadata":{"total_size":10},"weight_map":{"w":"a.safetensors","x":"a.safetensors","v":"b.safetensors"}}"#,
        )
        .unwrap();
        assert!(matches!(
            reconcile(&dir, &index),
            Err(StError::TotalSize { .. })
        ));
    }
}

fn read_at(file: &File, buffer: &mut [u8], offset: u64, path: &Path) -> Result<(), StError> {
    file.read_exact_at(buffer, offset)
        .map_err(|error| StError::Io {
            path: path.to_path_buf(),
            detail: error.to_string(),
        })
}

/// `h3_weight_store_open` (`h3_weights.c:44-114`): every `.safetensors`
/// file, sorted by path, headers parsed up front. Like the reference, this
/// scan keeps dot-files; `inventory` is the one that skips them.
pub fn open_dir(dir: &Path) -> Result<Store, StError> {
    let mut paths = safetensors_paths(dir, false)?;

    if paths.is_empty() {
        return Err(StError::NoShards(dir.to_path_buf()));
    }

    paths.sort();
    let mut shards = Vec::with_capacity(paths.len());
    for path in paths {
        shards.push(read_shard(&path)?);
    }

    Ok(Store { shards })
}

/// `h3_st_inventory_dir` (`h3_safetensors.c:540-585`).
pub fn inventory(dir: &Path) -> Result<Inventory, StError> {
    let paths = safetensors_paths(dir, true)?;

    if paths.is_empty() {
        return Err(StError::NoShards(dir.to_path_buf()));
    }

    let mut totals = Inventory::default();
    for path in paths {
        let shard = read_shard(&path)?;
        totals.files += 1;
        totals.bytes += shard.file_size;
        totals.tensors += shard.tensors.len() as u64;
        totals.tensor_bytes += shard.tensor_bytes();
    }

    Ok(totals)
}

fn safetensors_paths(dir: &Path, skip_dot_files: bool) -> Result<Vec<PathBuf>, StError> {
    let entries = fs::read_dir(dir).map_err(|error| StError::Io {
        path: dir.to_path_buf(),
        detail: error.to_string(),
    })?;

    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| StError::Io {
            path: dir.to_path_buf(),
            detail: error.to_string(),
        })?;
        let name = entry.file_name();
        let name = name.to_string_lossy();

        if skip_dot_files && name.starts_with('.') {
            continue;
        }
        if !name.ends_with(".safetensors") {
            continue;
        }
        paths.push(entry.path());
    }

    Ok(paths)
}

/// The HF shard index (`model.safetensors.index.json`): the model's own
/// declaration of tensor name to shard file, plus a total byte count.
#[derive(Clone, Debug)]
pub struct Index {
    pub total_size: Option<u64>,
    pub weight_map: BTreeMap<String, String>,
}

pub fn read_index(path: &Path) -> Result<Index, StError> {
    let text = fs::read_to_string(path).map_err(|error| StError::Io {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })?;

    let value: Value = serde_json::from_str(&text).map_err(|error| StError::Index {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })?;

    let object = value.as_object().ok_or_else(|| StError::Index {
        path: path.to_path_buf(),
        detail: "index is not a JSON object".into(),
    })?;

    let total_size = object
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("total_size"))
        .and_then(Value::as_u64);

    let weight_map = object
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| StError::Index {
            path: path.to_path_buf(),
            detail: "weight_map is missing".into(),
        })?;

    let mut map = BTreeMap::new();
    for (name, shard) in weight_map {
        let shard = shard.as_str().ok_or_else(|| StError::Index {
            path: path.to_path_buf(),
            detail: format!("{name}: shard name is not a string"),
        })?;
        map.insert(name.clone(), shard.to_string());
    }

    Ok(Index {
        total_size,
        weight_map: map,
    })
}

/// The reconciled two-sided manifest: what the index declares against what
/// the shard headers actually hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub shards: u64,
    pub tensors: u64,
    pub tensor_bytes: u64,
}

/// P3's gate: every index entry resolves to the shard it names, every header
/// tensor is indexed, and `total_size` matches the header byte total.
pub fn reconcile(dir: &Path, index_path: &Path) -> Result<Manifest, StError> {
    let store = open_dir(dir)?;
    let index = read_index(index_path)?;

    let shard_names: Vec<String> = store
        .shards()
        .iter()
        .filter_map(|shard| {
            shard
                .path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .collect();

    for (name, indexed_shard) in &index.weight_map {
        if !shard_names.contains(indexed_shard) {
            return Err(StError::Index {
                path: index_path.to_path_buf(),
                detail: format!("{name}: shard {indexed_shard} is not in the directory"),
            });
        }

        let (shard, _) = store.find(name).ok_or_else(|| StError::MissingTensor {
            name: name.clone(),
            shard: indexed_shard.clone(),
        })?;

        let actual = shard
            .path
            .file_name()
            .map(|file| file.to_string_lossy().into_owned())
            .unwrap_or_default();

        if actual != *indexed_shard {
            return Err(StError::WrongShard {
                name: name.clone(),
                indexed: indexed_shard.clone(),
                actual,
            });
        }
    }

    for shard in store.shards() {
        for tensor in &shard.tensors {
            if !index.weight_map.contains_key(&tensor.name) {
                return Err(StError::Unindexed(tensor.name.clone()));
            }
        }
    }

    let tensor_bytes = store.tensor_bytes();
    if let Some(expected) = index.total_size {
        if expected != tensor_bytes {
            return Err(StError::TotalSize {
                expected,
                got: tensor_bytes,
            });
        }
    }

    Ok(Manifest {
        shards: store.shards().len() as u64,
        tensors: store.tensor_count(),
        tensor_bytes,
    })
}
