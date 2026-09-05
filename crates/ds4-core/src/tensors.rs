//! Host-owned GGUF tensor directory + split-shard remap plan.
//!
//! Copied from `ds4.c` (`parse_tensors`, `tensor_nbytes`,
//! `model_split_sibling_path`, split concat). When installed, native
//! `model_open` skips `parse_tensors` and applies this table. CUDA
//! weight upload stays inside `ds4_engine_open`.

use std::path::{Path, PathBuf};

use crate::gguf::{GgufError, GgufFile};
use crate::mapped::page_size;

pub const MAX_DIMS: u32 = 8;

#[derive(Debug)]
pub enum TensorError {
    Gguf(GgufError),
    BadDims,
    Overflow,
    OutsideFile,
    SplitPath,
    SplitFirst,
    ExpectedTensors { declared: u32, got: u64 },
}

impl TensorError {
    pub fn token(&self) -> String {
        match self {
            TensorError::Gguf(e) => e.token(),
            TensorError::BadDims => "bad-dims".into(),
            TensorError::Overflow => "overflow".into(),
            TensorError::OutsideFile => "outside-file".into(),
            TensorError::SplitPath => "split-path".into(),
            TensorError::SplitFirst => "split-first".into(),
            TensorError::ExpectedTensors { declared, got } => {
                format!("split-tensors {declared}!={got}")
            }
        }
    }
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for TensorError {}

impl From<GgufError> for TensorError {
    fn from(e: GgufError) -> Self {
        TensorError::Gguf(e)
    }
}

/// C `gguf_types[]`. Holes (4, 5) are unnamed / unsupported.
const GGUF_TYPES: [Option<(&'static str, u32, u32)>; 31] = [
    Some(("f32", 1, 4)),
    Some(("f16", 1, 2)),
    Some(("q4_0", 32, 18)),
    Some(("q4_1", 32, 20)),
    None,
    None,
    Some(("q5_0", 32, 22)),
    Some(("q5_1", 32, 24)),
    Some(("q8_0", 32, 34)),
    Some(("q8_1", 32, 40)),
    Some(("q2_k", 256, 84)),
    Some(("q3_k", 256, 110)),
    Some(("q4_k", 256, 144)),
    Some(("q5_k", 256, 176)),
    Some(("q6_k", 256, 210)),
    Some(("q8_k", 256, 292)),
    Some(("iq2_xxs", 256, 66)),
    Some(("iq2_xs", 256, 74)),
    Some(("iq3_xxs", 256, 98)),
    Some(("iq1_s", 256, 110)),
    Some(("iq4_nl", 256, 50)),
    Some(("iq3_s", 256, 110)),
    Some(("iq2_s", 256, 82)),
    Some(("iq4_xs", 256, 136)),
    Some(("i8", 1, 1)),
    Some(("i16", 1, 2)),
    Some(("i32", 1, 4)),
    Some(("i64", 1, 8)),
    Some(("f64", 1, 8)),
    Some(("iq1_m", 256, 56)),
    Some(("bf16", 1, 2)),
];

pub fn tensor_type_name(typ: u32) -> &'static str {
    GGUF_TYPES
        .get(typ as usize)
        .and_then(|t| t.map(|(n, _, _)| n))
        .unwrap_or("unknown")
}

/// C `tensor_nbytes`. Unsupported types return None (C leaves bytes=0).
pub fn tensor_nbytes(typ: u32, elements: u64) -> Option<u64> {
    let (_, block_elems, block_bytes) = *GGUF_TYPES.get(typ as usize)?.as_ref()?;
    if block_elems == 0 {
        return None;
    }
    let blocks = elements.saturating_add(u64::from(block_elems) - 1) / u64::from(block_elems);
    blocks.checked_mul(u64::from(block_bytes))
}

pub fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + alignment - rem
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ndim: u32,
    pub dim: [u64; MAX_DIMS as usize],
    pub typ: u32,
    pub rel_offset: u64,
    pub abs_offset: u64,
    pub elements: u64,
    pub bytes: u64,
    pub shard: u32,
}

#[derive(Debug, Clone)]
pub struct ShardPlan {
    pub path: PathBuf,
    pub size: u64,
    pub base: u64,
}

#[derive(Debug, Clone)]
pub struct TensorInventory {
    pub shards: Vec<ShardPlan>,
    pub tensors: Vec<TensorInfo>,
    pub data_pos: u64,
    pub alignment: u64,
    pub page: u64,
}

impl TensorInfo {
    pub fn dump_line(&self) -> String {
        let mut dims = String::new();
        for i in 0..self.ndim as usize {
            if i > 0 {
                dims.push(',');
            }
            dims.push_str(&self.dim[i].to_string());
        }
        format!(
            "T {} ndim={} dims={} type={}({}) elems={} bytes={} rel={} abs={} shard={}",
            self.name,
            self.ndim,
            dims,
            self.typ,
            tensor_type_name(self.typ),
            self.elements,
            self.bytes,
            self.rel_offset,
            self.abs_offset,
            self.shard
        )
    }
}

impl TensorInventory {
    /// C `model_find_tensor`: first exact name match.
    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn find_index(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    pub fn dump(&self) -> String {
        let mut out = format!(
            "DATA_POS {} ALIGN {} PAGE {} SHARDS {} N {}\n",
            self.data_pos,
            self.alignment,
            self.page,
            self.shards.len(),
            self.tensors.len()
        );
        for s in &self.shards {
            out.push_str(&format!(
                "SHARD {} size={} base={}\n",
                s.path.display(),
                s.size,
                s.base
            ));
        }
        for t in &self.tensors {
            out.push_str(&t.dump_line());
            out.push('\n');
        }
        out
    }
}

fn parse_one(
    g: &GgufFile,
    shard: u32,
    file_size: u64,
) -> Result<(u64, Vec<TensorInfo>), TensorError> {
    let data = g.as_bytes();
    let mut pos = g.tensor_dir_pos;
    let mut tensors = Vec::with_capacity(g.n_tensors.min(1024) as usize);
    for _ in 0..g.n_tensors {
        if pos + 8 > data.len() {
            return Err(TensorError::Gguf(GgufError::Truncated));
        }
        let nlen = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let n = usize::try_from(nlen).map_err(|_| TensorError::Gguf(GgufError::Truncated))?;
        if pos + n > data.len() {
            return Err(TensorError::Gguf(GgufError::Truncated));
        }
        let name = String::from_utf8_lossy(&data[pos..pos + n]).into_owned();
        pos += n;
        if pos + 4 > data.len() {
            return Err(TensorError::Gguf(GgufError::Truncated));
        }
        let ndim = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        if ndim == 0 || ndim > MAX_DIMS {
            return Err(TensorError::BadDims);
        }
        let mut dim = [0u64; MAX_DIMS as usize];
        let mut elements = 1u64;
        for d in 0..ndim as usize {
            if pos + 8 > data.len() {
                return Err(TensorError::Gguf(GgufError::Truncated));
            }
            dim[d] = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            if dim[d] != 0 && elements > u64::MAX / dim[d] {
                return Err(TensorError::Overflow);
            }
            elements *= dim[d];
        }
        if pos + 4 + 8 > data.len() {
            return Err(TensorError::Gguf(GgufError::Truncated));
        }
        let typ = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let rel_offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let bytes = tensor_nbytes(typ, elements).unwrap_or(0);
        tensors.push(TensorInfo {
            name,
            ndim,
            dim,
            typ,
            rel_offset,
            abs_offset: 0,
            elements,
            bytes,
            shard,
        });
    }
    let data_pos = align_up(pos as u64, g.alignment);
    for t in &mut tensors {
        t.abs_offset = data_pos
            .checked_add(t.rel_offset)
            .ok_or(TensorError::Overflow)?;
        if t.bytes != 0 && (t.abs_offset > file_size || t.bytes > file_size - t.abs_offset) {
            return Err(TensorError::OutsideFile);
        }
    }
    Ok((data_pos, tensors))
}

fn split_sibling_path_c(path: &str, index: u32, count: u32) -> Option<String> {
    // C: snprintf("%.*s%05u-of-%05u.gguf", prefix, index+1, count)
    let dash = path.rfind('-')?;
    let tail = &path[dash..];
    // sscanf(dash, "-%05u.gguf", &parsed_count)
    if !tail.starts_with('-') || !tail.ends_with(".gguf") {
        return None;
    }
    let num_part = &tail[1..tail.len() - 5];
    if num_part.len() != 5 || !num_part.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let parsed_count: u32 = num_part.parse().ok()?;
    if parsed_count != count {
        return None;
    }
    if dash < 9 {
        return None;
    }
    let of = dash - 3;
    if &path[of..dash] != "-of" {
        return None;
    }
    let num = of - 5;
    if num < 1 || path.as_bytes().get(num - 1) != Some(&b'-') {
        return None;
    }
    if !path.as_bytes()[num..of].iter().all(|c| c.is_ascii_digit()) || of - num != 5 {
        return None;
    }
    let prefix = &path[..num];
    Some(format!("{prefix}{:05}-of-{:05}.gguf", index + 1, count))
}

/// Public C-matching sibling remap.
pub fn model_split_sibling_path(path: &str, index: u32, count: u32) -> Option<String> {
    split_sibling_path_c(path, index, count)
}

fn file_len(path: &Path) -> Result<u64, TensorError> {
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| TensorError::Gguf(GgufError::Io(e)))
}

impl TensorInventory {
    pub fn open(path: &Path) -> Result<Self, TensorError> {
        let g = GgufFile::open(path)?;
        Self::from_file(path, &g)
    }

    pub fn from_file(path: &Path, g: &GgufFile) -> Result<Self, TensorError> {
        let page = page_size();
        let path_s = path.to_string_lossy();
        let split = g.split_count();
        if split <= 1 {
            let size = file_len(path)?;
            let (data_pos, tensors) = parse_one(g, 0, size)?;
            return Ok(Self {
                shards: vec![ShardPlan {
                    path: path.to_path_buf(),
                    size,
                    base: 0,
                }],
                tensors,
                data_pos,
                alignment: g.alignment,
                page,
            });
        }
        if split > 99_999 {
            return Err(TensorError::SplitPath);
        }
        let first = model_split_sibling_path(&path_s, 0, split).ok_or(TensorError::SplitPath)?;
        if first != path_s {
            return Err(TensorError::SplitFirst);
        }
        let mut shards = Vec::with_capacity(split as usize);
        let mut total = 0u64;
        for k in 0..split {
            let shard_path = if k == 0 {
                path.to_path_buf()
            } else {
                PathBuf::from(
                    model_split_sibling_path(&path_s, k, split).ok_or(TensorError::SplitPath)?,
                )
            };
            let size = file_len(&shard_path)?;
            if size < 32 {
                return Err(TensorError::Gguf(GgufError::TooSmall));
            }
            shards.push(ShardPlan {
                path: shard_path,
                size,
                base: total,
            });
            let span = align_up(size, page);
            total = total.checked_add(span).ok_or(TensorError::Overflow)?;
        }

        let mut all = Vec::new();
        let mut data_pos = 0u64;
        for (k, shard) in shards.iter().enumerate() {
            let gf = if k == 0 {
                None
            } else {
                Some(GgufFile::open(&shard.path)?)
            };
            let file = if k == 0 { g } else { gf.as_ref().unwrap() };
            let (dp, mut ts) = parse_one(file, k as u32, shard.size)?;
            if k == 0 {
                data_pos = dp;
            }
            for t in &mut ts {
                t.abs_offset = t
                    .abs_offset
                    .checked_add(shard.base)
                    .ok_or(TensorError::Overflow)?;
            }
            all.extend(ts);
        }

        if let Some(declared) = expected_split_tensors(g) {
            if declared != 0 && u64::from(declared) != all.len() as u64 {
                return Err(TensorError::ExpectedTensors {
                    declared,
                    got: all.len() as u64,
                });
            }
        }

        Ok(Self {
            shards,
            tensors: all,
            data_pos,
            alignment: g.alignment,
            page,
        })
    }
}

fn expected_split_tensors(g: &GgufFile) -> Option<u32> {
    // C reads INT32 or UINT32 as four LE bytes into uint32_t.
    if let Some(v) = g.get_u32("split.tensors.count") {
        return Some(v);
    }
    let e = g
        .kv_entries()
        .iter()
        .find(|e| g.key_bytes(e) == b"split.tensors.count")?;
    if e.typ != crate::gguf::GGUF_VALUE_INT32 {
        return None;
    }
    let b = g.as_bytes().get(e.value_pos..e.value_pos + 4)?;
    Some(u32::from_le_bytes(b.try_into().ok()?))
}

pub fn dump_nbytes_table() -> String {
    let mut out = String::new();
    for typ in 0..=30u32 {
        for elems in [1u64, 31, 32, 256, 257] {
            match tensor_nbytes(typ, elems) {
                Some(b) => out.push_str(&format!(
                    "NBYTES type={} name={} elems={} bytes={}\n",
                    typ,
                    tensor_type_name(typ),
                    elems,
                    b
                )),
                None => out.push_str(&format!(
                    "NBYTES type={} name={} elems={} FAIL\n",
                    typ,
                    tensor_type_name(typ),
                    elems
                )),
            }
        }
    }
    out
}

/// C `ds4_host_tensor_dir_consume`. `host_tensors = None` with `host_n > 0`
/// is `tensors-null`. `host = None` (absent dir) is the C default no-op.
pub fn consume_host_dir(
    native: &mut [TensorInfo],
    native_data_pos: u64,
    native_alignment: u64,
    host: Option<(&[TensorInfo], u64, u64)>,
    host_n_override: Option<u32>,
    host_slots_null: bool,
) -> Result<(), &'static str> {
    let Some((host_rows, host_data_pos, host_alignment)) = host else {
        return Ok(());
    };
    let n = host_n_override.unwrap_or(host_rows.len() as u32);
    if host_slots_null && n > 0 {
        return Err("tensors-null");
    }
    if n as usize != native.len() {
        return Err("count-mismatch");
    }
    for i in 0..native.len() {
        let h = &host_rows[i];
        let ntv = &mut native[i];
        if h.name.is_empty() {
            return Err("name-empty");
        }
        if ntv.name != h.name {
            return Err("name-mismatch");
        }
        if ntv.typ != h.typ {
            return Err("type-mismatch");
        }
        if ntv.ndim != h.ndim || h.ndim == 0 || h.ndim > MAX_DIMS {
            return Err("dim-mismatch");
        }
        if ntv.dim[..h.ndim as usize] != h.dim[..h.ndim as usize] {
            return Err("dim-mismatch");
        }
        if ntv.rel_offset != h.rel_offset || ntv.abs_offset != h.abs_offset {
            return Err("offset-mismatch");
        }
        if ntv.bytes != h.bytes {
            return Err("bytes-mismatch");
        }
        ntv.rel_offset = h.rel_offset;
        ntv.abs_offset = h.abs_offset;
        ntv.bytes = h.bytes;
    }
    if host_data_pos != native_data_pos || host_alignment != native_alignment {
        return Err("data-mismatch");
    }
    Ok(())
}

/// C `ds4_host_tensor_dir_apply`. `host = None` is `dir-null`.
/// `out = None` with `n > 0` is `out-null`.
pub fn apply_host_dir(
    out: Option<&mut [TensorInfo]>,
    cap: u32,
    host: Option<&[TensorInfo]>,
    host_n_override: Option<u32>,
    host_slots_null: bool,
) -> Result<(), &'static str> {
    let Some(host_rows) = host else {
        return Err("dir-null");
    };
    let n = host_n_override.unwrap_or(host_rows.len() as u32);
    if host_slots_null && n > 0 {
        return Err("tensors-null");
    }
    if n != cap {
        return Err("count-mismatch");
    }
    let Some(out) = out else {
        return if n > 0 { Err("out-null") } else { Ok(()) };
    };
    if out.len() as u32 != cap {
        return Err("count-mismatch");
    }
    for i in 0..n as usize {
        let h = &host_rows[i];
        if h.name.is_empty() {
            return Err("name-empty");
        }
        if h.ndim == 0 || h.ndim > MAX_DIMS {
            return Err("dim-mismatch");
        }
        out[i] = TensorInfo {
            name: h.name.clone(),
            ndim: h.ndim,
            dim: h.dim,
            typ: h.typ,
            rel_offset: h.rel_offset,
            abs_offset: h.abs_offset,
            elements: h.elements,
            bytes: h.bytes,
            shard: h.shard,
        };
    }
    Ok(())
}

fn consume_sample(name: &str) -> TensorInfo {
    let mut dim = [0u64; MAX_DIMS as usize];
    dim[0] = 4;
    dim[1] = 8;
    TensorInfo {
        name: name.into(),
        ndim: 2,
        dim,
        typ: 0,
        rel_offset: 0,
        abs_offset: 32,
        elements: 32,
        bytes: 128,
        shard: 0,
    }
}

fn consume_token(
    native: &mut [TensorInfo],
    ndp: u64,
    na: u64,
    host: Option<(&[TensorInfo], u64, u64)>,
    host_n_override: Option<u32>,
    host_slots_null: bool,
) -> String {
    match consume_host_dir(native, ndp, na, host, host_n_override, host_slots_null) {
        Ok(()) => "ok".into(),
        Err(e) => e.into(),
    }
}

/// Fixed C↔Rust consume tapes (same cases as `load_c_oracle consume-tapes`).
pub fn dump_consume_tapes() -> String {
    let mut out = String::new();
    let a = consume_sample("tok_embd");
    let b = consume_sample("output");
    let host = [a.clone(), b.clone()];
    let mut native = host.clone();

    out.push_str(&format!(
        "absent {}\n",
        consume_token(&mut native, 32, 32, None, None, false)
    ));
    native = host.clone();
    out.push_str(&format!(
        "ok {}\n",
        consume_token(&mut native, 32, 32, Some((&host, 32, 32)), None, false)
    ));
    native = host.clone();
    out.push_str(&format!(
        "tensors-null {}\n",
        consume_token(&mut native, 32, 32, Some((&host, 32, 32)), Some(2), true)
    ));
    native = host.clone();
    out.push_str(&format!(
        "count {}\n",
        consume_token(&mut native, 32, 32, Some((&host[..1], 32, 32)), None, false)
    ));
    let mut bad_name = host.clone();
    bad_name[1].name = "other".into();
    native = host.clone();
    out.push_str(&format!(
        "name {}\n",
        consume_token(&mut native, 32, 32, Some((&bad_name, 32, 32)), None, false)
    ));
    let mut empty = host.clone();
    empty[0].name.clear();
    native = host.clone();
    out.push_str(&format!(
        "name-empty {}\n",
        consume_token(&mut native, 32, 32, Some((&empty, 32, 32)), None, false)
    ));
    let mut bad_ty = host.clone();
    bad_ty[0].typ = 1;
    native = host.clone();
    out.push_str(&format!(
        "type {}\n",
        consume_token(&mut native, 32, 32, Some((&bad_ty, 32, 32)), None, false)
    ));
    let mut bad_dim = host.clone();
    bad_dim[0].dim[0] = 2;
    native = host.clone();
    out.push_str(&format!(
        "dim {}\n",
        consume_token(&mut native, 32, 32, Some((&bad_dim, 32, 32)), None, false)
    ));
    let mut bad_off = host.clone();
    bad_off[0].rel_offset = 64;
    native = host.clone();
    out.push_str(&format!(
        "offset {}\n",
        consume_token(&mut native, 32, 32, Some((&bad_off, 32, 32)), None, false)
    ));
    let mut bad_b = host.clone();
    bad_b[0].bytes = 64;
    native = host.clone();
    out.push_str(&format!(
        "bytes {}\n",
        consume_token(&mut native, 32, 32, Some((&bad_b, 32, 32)), None, false)
    ));
    native = host.clone();
    out.push_str(&format!(
        "data {}\n",
        consume_token(&mut native, 32, 32, Some((&host, 64, 32)), None, false)
    ));
    out
}

fn apply_token(
    out: Option<&mut [TensorInfo]>,
    cap: u32,
    host: Option<&[TensorInfo]>,
    host_n_override: Option<u32>,
    host_slots_null: bool,
) -> String {
    match apply_host_dir(out, cap, host, host_n_override, host_slots_null) {
        Ok(()) => "ok".into(),
        Err(e) => e.into(),
    }
}

/// Fixed C↔Rust apply tapes (same cases as `load_c_oracle apply-tapes`).
pub fn dump_apply_tapes() -> String {
    let mut out = String::new();
    let a = consume_sample("tok_embd");
    let b = consume_sample("output");
    let host = [a.clone(), b.clone()];
    let mut native = [consume_sample("x"), consume_sample("y")];

    out.push_str(&format!(
        "dir-null {}\n",
        apply_token(Some(&mut native), 2, None, None, false)
    ));
    out.push_str(&format!(
        "tensors-null {}\n",
        apply_token(Some(&mut native), 2, Some(&host), Some(2), true)
    ));
    out.push_str(&format!(
        "count {}\n",
        apply_token(Some(&mut native), 2, Some(&host[..1]), None, false)
    ));
    out.push_str(&format!(
        "out-null {}\n",
        apply_token(None, 2, Some(&host), None, false)
    ));
    let mut empty = host.clone();
    empty[0].name.clear();
    out.push_str(&format!(
        "name-empty {}\n",
        apply_token(Some(&mut native), 2, Some(&empty), None, false)
    ));
    let mut bad_dim = host.clone();
    bad_dim[0].ndim = 0;
    out.push_str(&format!(
        "dim {}\n",
        apply_token(Some(&mut native), 2, Some(&bad_dim), None, false)
    ));
    native = [consume_sample("x"), consume_sample("y")];
    out.push_str(&format!(
        "ok {}\n",
        apply_token(Some(&mut native), 2, Some(&host), None, false)
    ));
    out.push_str(&format!(
        "row0 {} {} {} {} {} {}\n",
        native[0].name,
        native[0].ndim,
        native[0].typ,
        native[0].rel_offset,
        native[0].abs_offset,
        native[0].bytes
    ));
    out.push_str(&format!(
        "row1 {} {} {} {} {} {}\n",
        native[1].name,
        native[1].ndim,
        native[1].typ,
        native[1].rel_offset,
        native[1].abs_offset,
        native[1].bytes
    ));
    out.push_str(&format!(
        "then-consume {}\n",
        consume_token(&mut native, 32, 32, Some((&host, 32, 32)), None, false)
    ));
    out
}

pub fn dump_sibling_script(path: &str, index: u32, count: u32) -> String {
    match model_split_sibling_path(path, index, count) {
        Some(p) => format!("SIBLING {p}\n"),
        None => "SIBLING FAIL\n".into(),
    }
}
