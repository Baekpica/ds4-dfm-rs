//! CPU decode of one release entry into a host tensor.
//!
//! The release states each entry's shape, container geometry and payload
//! layout; this module reads the file, re-checks the header against the
//! release, and sums the residual stages into one row-major f32 matrix. The
//! value rule and every field layout come from [`crate::vq`], whose header
//! cites the two producers (`/data/YoungAi/vq_fmt.h` for the engine, and
//! `/data/VQ_tools/crates/vq-format/src/lib.rs` plus
//! `/data/VQ_tools/c/gpu/build-vq-gpu.c` for the tools that wrote the
//! artifact).
//!
//! What this does not do: it holds no device state, computes no forward pass
//! and does not keep a decoded copy. A caller that needs a tensor gets the
//! whole matrix as f32 and owns it.

use std::path::Path;

use super::release::{Codebook, Entry, Payload};
use super::Error;
use crate::vq::VqMatrix;

const RQVQ: &[u8; 4] = b"RQVQ";
const DQVQ: &[u8; 4] = b"DQVQ";
const DQV3: &[u8; 4] = b"DQV3";
const RQVF: &[u8; 4] = b"RQVF";
/// `vq-format`'s codebook format word for the f16 codebook.
const CB_F16: u32 = 0;
/// `vq-format`'s codebook format word for the one-byte E4M3FN codebook.
const CB_E4M3: u32 = 0x334d_3445;

/// A decoded tensor, row-major f32.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    /// Value at `row`, `col`.
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }
}

fn u16le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64le(b: &[u8], at: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(w)
}

/// Decode one entry of an opened release.
///
/// A copied entry is an `RQVF` sidecar read as f32; an encoded entry is an
/// `RQVQ` container whose `stages * groups` payloads are summed into the
/// tensor at their group's first row
/// (`/data/VQ_tools/crates/vq-format/src/lib.rs:483-560`).
pub fn decode(dir: &Path, entry: &Entry) -> Result<Tensor, Error> {
    let path = dir.join(format!("{}.bin", entry.name));
    let file = std::fs::read(&path).map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
    if entry.encoded {
        decode_container(&path, &file, entry)
    } else {
        decode_sidecar(&path, &file, entry)
    }
}

fn decode_sidecar(path: &Path, file: &[u8], entry: &Entry) -> Result<Tensor, Error> {
    let what = path.display();
    if file.len() < 12 || &file[0..4] != RQVF {
        return Err(Error::new(format!("{what}: not an RQVF sidecar")));
    }
    let rows = u32le(file, 4) as usize;
    let cols = u32le(file, 8) as usize;
    if rows != entry.rows || cols != entry.cols {
        return Err(Error::new(format!(
            "{what}: sidecar is {rows}x{cols}, release says {}x{}",
            entry.rows, entry.cols
        )));
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| Error::new(format!("{what}: shape overflows")))?;
    if file.len() != 12 + count * 4 {
        return Err(Error::new(format!(
            "{what}: {} bytes, {rows}x{cols} f32 is {}",
            file.len(),
            12 + count * 4
        )));
    }
    let data = file[12..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Tensor { rows, cols, data })
}

fn decode_container(path: &Path, file: &[u8], entry: &Entry) -> Result<Tensor, Error> {
    let what = path.display();
    if file.len() < 36 || &file[0..4] != RQVQ {
        return Err(Error::new(format!("{what}: not an RQVQ container")));
    }
    let dim = u16le(file, 4) as u32;
    let nc = u16le(file, 6) as u32;
    let stages = u16le(file, 8) as usize;
    let groups = match u16le(file, 10) as usize {
        0 | 1 => 1,
        g => g,
    };
    let rows = u32le(file, 12) as usize;
    let cols = u32le(file, 16) as usize;
    let format = u32le(file, 20);
    let payload = u64le(file, 24) as usize;
    if rows != entry.rows || cols != entry.cols {
        return Err(Error::new(format!(
            "{what}: header is {rows}x{cols}, release says {}x{}",
            entry.rows, entry.cols
        )));
    }
    for (field, header, recorded) in [
        ("dim", dim as usize, entry.dim),
        ("nc", nc as usize, entry.nc),
        ("stages", stages, entry.stages),
        ("groups", groups, entry.groups),
    ] {
        if recorded != Some(header) {
            return Err(Error::new(format!(
                "{what}: header {field} is {header}, release says {recorded:?}"
            )));
        }
    }
    let cb_width = match format {
        CB_F16 => 2,
        CB_E4M3 => 1,
        other => {
            return Err(Error::new(format!(
                "{what}: unknown codebook format {other:#010x}"
            )))
        }
    };
    let recorded = entry
        .codebook
        .ok_or_else(|| Error::new(format!("{what}: release records no codebook")))?;
    let expected = if cb_width == 2 {
        Codebook::F16
    } else {
        Codebook::E4M3
    };
    if recorded != expected {
        return Err(Error::new(format!(
            "{what}: header codebook {expected:?}, release says {recorded:?}"
        )));
    }
    if dim == 0 || nc == 0 || stages == 0 || rows == 0 || cols == 0 {
        return Err(Error::new(format!("{what}: header states a zero field")));
    }
    if rows % groups != 0 || cols % dim as usize != 0 {
        return Err(Error::new(format!(
            "{what}: {rows}x{cols} does not split into {groups} groups of dim {dim}"
        )));
    }
    let kind = if &file[32..36] == DQV3 {
        true
    } else if &file[32..36] == DQVQ {
        false
    } else {
        return Err(Error::new(format!(
            "{what}: first payload is neither DQVQ nor DQV3"
        )));
    };
    let recorded = entry
        .payload
        .ok_or_else(|| Error::new(format!("{what}: release records no payload kind")))?;
    if recorded != if kind { Payload::V3 } else { Payload::V2 } {
        return Err(Error::new(format!(
            "{what}: payload kind {recorded:?} disagrees with the file"
        )));
    }
    let payloads = stages * groups;
    if file.len() != 32 + payload {
        return Err(Error::new(format!(
            "{what}: file is {} bytes, header says {}",
            file.len(),
            32 + payload
        )));
    }
    if payload % payloads != 0 {
        return Err(Error::new(format!(
            "{what}: {payload} payload bytes do not divide into {payloads} payloads"
        )));
    }
    let per = payload / payloads;
    let group_rows = rows / groups;
    let mut data = vec![0f32; rows * cols];
    for p in 0..payloads {
        let at = 32 + p * per;
        let matrix = VqMatrix::from_payload(
            &file[at..at + per],
            cb_width,
            dim as u16,
            group_rows as u32,
            cols as u32,
        )
        .map_err(|e| Error::new(format!("{what}: payload {p}: {e}")))?;
        if matrix.nc != nc {
            return Err(Error::new(format!(
                "{what}: payload {p} states nc {}, header {nc}",
                matrix.nc
            )));
        }
        matrix.add_into(&mut data, (p % groups) * group_rows);
    }
    Ok(Tensor { rows, cols, data })
}
