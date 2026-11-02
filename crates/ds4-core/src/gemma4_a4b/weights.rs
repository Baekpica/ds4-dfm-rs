//! Streaming matrix access for the CPU forward.
//!
//! A decoded `gate_up_proj` is 180224 x 2816 f32 (2 GB per layer), so the
//! forward never materialises a matrix. It maps the container file, keeps the
//! header [`decode`] already checked, and either dots every row (a projection)
//! or gathers the handful of rows a routed expert needs.
//!
//! The arithmetic is not new: every entry point here builds the same
//! [`VqMatrix`] the decode gate used and calls its row dot, so the value rule
//! and the stage sum keep the one implementation
//! (`/data/VQ_tools/crates/vq-format/src/lib.rs:445-478`).
//!
//! The mapping is read-only and private; the page cache is the only copy, so
//! the nine-gigabyte artifact can be walked on a host that cannot hold its
//! decoded form.

use std::path::{Path, PathBuf};

use super::decode::{self, Header, Tensor};
use super::release::Entry;
use super::Error;
use crate::mapped::MappedFile;
use crate::vq::VqMatrix;

/// The payload offset of an `RQVQ` container (`vq-format/src/lib.rs:314-361`).
const PAYLOAD_AT: usize = 32;

/// How many worker threads a matrix walk may use.
pub fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// One weight matrix: a mapped `RQVQ` container or a decoded f32 sidecar.
pub enum Matrix {
    Vq(Vq),
    Dense(Tensor),
}

pub struct Vq {
    path: PathBuf,
    h: Header,
    map: MappedFile,
}

impl Matrix {
    /// Open one release entry. A container is mapped and its header checked
    /// against the release; a copied sidecar is decoded in full (the forward
    /// only does that for small tensors).
    pub fn open(dir: &Path, entry: &Entry) -> Result<Matrix, Error> {
        if !entry.encoded {
            return Ok(Matrix::Dense(decode::decode(dir, entry)?));
        }
        let path = dir.join(format!("{}.bin", entry.name));
        let map = MappedFile::open_ro(&path)
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        let h = decode::header(&path, map.as_slice(), entry)?;
        let matrix = Vq { path, h, map };
        // One parse of every payload now, so a thread that walks the rows has
        // nothing left to refuse.
        for p in 0..matrix.h.count() {
            matrix.matrix(p)?;
        }
        Ok(Matrix::Vq(matrix))
    }

    pub fn rows(&self) -> usize {
        match self {
            Matrix::Vq(v) => v.h.rows,
            Matrix::Dense(t) => t.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Matrix::Vq(v) => v.h.cols,
            Matrix::Dense(t) => t.cols,
        }
    }

    /// `x` dotted with every row: the projection `W x`.
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>, Error> {
        match self {
            Matrix::Vq(v) => v.matvec(x),
            Matrix::Dense(t) => {
                if x.len() != t.cols {
                    return Err(Error::new(format!(
                        "matvec: x has {} values, the matrix has {} columns",
                        x.len(),
                        t.cols
                    )));
                }
                let mut out = vec![0f32; t.rows];
                for (row, slot) in out.iter_mut().enumerate() {
                    let mut s = 0.0f32;
                    for (col, value) in x.iter().enumerate() {
                        s += t.data[row * t.cols + col] * value;
                    }
                    *slot = s;
                }
                Ok(out)
            }
        }
    }

    /// `x` dotted with the named rows, in the order given. A routed expert
    /// gathers its 1408 gate/up rows or its 2816 down rows this way instead
    /// of walking the whole stack.
    pub fn dot_rows(&self, rows: &[usize], x: &[f32]) -> Result<Vec<f32>, Error> {
        match self {
            Matrix::Vq(v) => v.dot_rows(rows, x),
            Matrix::Dense(t) => {
                if x.len() != t.cols {
                    return Err(Error::new(format!(
                        "dot_rows: x has {} values, the matrix has {} columns",
                        x.len(),
                        t.cols
                    )));
                }
                let mut out = vec![0f32; rows.len()];
                for (slot, &row) in out.iter_mut().zip(rows) {
                    if row >= t.rows {
                        return Err(Error::new(format!(
                            "dot_rows: row {row} is outside {}",
                            t.rows
                        )));
                    }
                    let mut s = 0.0f32;
                    for (col, value) in x.iter().enumerate() {
                        s += t.data[row * t.cols + col] * value;
                    }
                    *slot = s;
                }
                Ok(out)
            }
        }
    }

    /// One row, decoded.
    pub fn row(&self, row: usize) -> Result<Vec<f32>, Error> {
        match self {
            Matrix::Vq(v) => v.row(row),
            Matrix::Dense(t) => {
                if row >= t.rows {
                    return Err(Error::new(format!("row {row} is outside {}", t.rows)));
                }
                Ok(t.data[row * t.cols..(row + 1) * t.cols].to_vec())
            }
        }
    }
}

impl Vq {
    fn payload(&self, p: usize) -> &[u8] {
        let at = PAYLOAD_AT + p * self.h.per;
        &self.map.as_slice()[at..at + self.h.per]
    }

    fn matrix(&self, p: usize) -> Result<VqMatrix<'_>, Error> {
        decode::matrix_of(&self.path, &self.h, self.payload(p), p)
    }

    fn matvec(&self, x: &[f32]) -> Result<Vec<f32>, Error> {
        if x.len() != self.h.cols {
            return Err(Error::new(format!(
                "matvec: x has {} values, the matrix has {} columns",
                x.len(),
                self.h.cols
            )));
        }
        let group_rows = self.h.group_rows();
        let groups = self.h.groups;
        let count = self.h.count();
        // Each stage has its own codebook; decode them once to f32 and share
        // them read-only with the row walkers.
        let mut codebooks = Vec::with_capacity(count);
        for p in 0..count {
            codebooks.push(self.matrix(p)?.codebook_f32());
        }
        let mut out = vec![0f32; self.h.rows];
        let threads = worker_count().min(out.len().max(1));
        if threads <= 1 || out.len() < 512 {
            for p in 0..count {
                let m = self.matrix(p)?;
                let base = (p % groups) * group_rows;
                for r in 0..group_rows {
                    out[base + r] += m.dot_row_codebook(r, x, &codebooks[p]);
                }
            }
            return Ok(out);
        }
        let chunk = out.len().div_ceil(threads);
        let mut results: Vec<Result<(), Error>> = Vec::with_capacity(threads);
        let codebooks = &codebooks;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for (i, part) in out.chunks_mut(chunk).enumerate() {
                let row0 = i * chunk;
                handles.push(scope.spawn(move || -> Result<(), Error> {
                    for p in 0..count {
                        let base = (p % groups) * group_rows;
                        let lo = row0.max(base);
                        let hi = (row0 + part.len()).min(base + group_rows);
                        if lo >= hi {
                            continue;
                        }
                        let m = self.matrix(p)?;
                        for r in lo..hi {
                            part[r - row0] +=
                                m.dot_row_codebook(r - base, x, &codebooks[p]);
                        }
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                match handle.join() {
                    Ok(result) => results.push(result),
                    Err(_) => results.push(Err(Error::new("matvec worker panicked"))),
                }
            }
        });
        for result in results {
            result?;
        }
        Ok(out)
    }

    fn dot_rows(&self, rows: &[usize], x: &[f32]) -> Result<Vec<f32>, Error> {
        if x.len() != self.h.cols {
            return Err(Error::new(format!(
                "dot_rows: x has {} values, the matrix has {} columns",
                x.len(),
                self.h.cols
            )));
        }
        for &row in rows {
            if row >= self.h.rows {
                return Err(Error::new(format!(
                    "dot_rows: row {row} is outside {}",
                    self.h.rows
                )));
            }
        }
        let group_rows = self.h.group_rows();
        let groups = self.h.groups;
        let stages = self.h.stages;
        let count = self.h.count();
        let mut codebooks = Vec::with_capacity(count);
        for p in 0..count {
            codebooks.push(self.matrix(p)?.codebook_f32());
        }
        let mut out = vec![0f32; rows.len()];
        let threads = worker_count().min(out.len().max(1));
        if threads <= 1 || out.len() < 256 {
            for (slot, &row) in out.iter_mut().zip(rows) {
                let group = row / group_rows;
                let mut sum = 0.0f32;
                for stage in 0..stages {
                    let p = stage * groups + group;
                    let m = self.matrix(p)?;
                    sum += m.dot_row_codebook(row - group * group_rows, x, &codebooks[p]);
                }
                *slot = sum;
            }
            return Ok(out);
        }
        let chunk = out.len().div_ceil(threads);
        let mut results: Vec<Result<(), Error>> = Vec::with_capacity(threads);
        let codebooks = &codebooks;
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for (part, row_part) in out.chunks_mut(chunk).zip(rows.chunks(chunk)) {
                handles.push(scope.spawn(move || -> Result<(), Error> {
                    let mut matrices: Vec<Option<VqMatrix<'_>>> =
                        (0..count).map(|_| None).collect();
                    for (slot, &row) in part.iter_mut().zip(row_part) {
                        let group = row / group_rows;
                        let mut sum = 0.0f32;
                        for stage in 0..stages {
                            let p = stage * groups + group;
                            if matrices[p].is_none() {
                                matrices[p] = Some(self.matrix(p)?);
                            }
                            let m = matrices[p].as_ref().expect("parsed above");
                            sum += m.dot_row_codebook(
                                row - group * group_rows,
                                x,
                                &codebooks[p],
                            );
                        }
                        *slot = sum;
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                match handle.join() {
                    Ok(result) => results.push(result),
                    Err(_) => results.push(Err(Error::new("dot_rows worker panicked"))),
                }
            }
        });
        for result in results {
            result?;
        }
        Ok(out)
    }

    fn row(&self, row: usize) -> Result<Vec<f32>, Error> {
        if row >= self.h.rows {
            return Err(Error::new(format!(
                "row {row} is outside {}",
                self.h.rows
            )));
        }
        let group_rows = self.h.group_rows();
        let group = row / group_rows;
        let mut out = vec![0f32; self.h.cols];
        let mut word = vec![0f32; usize::from(self.h.dim)];
        for stage in 0..self.h.stages {
            let p = stage * self.h.groups + group;
            let m = self.matrix(p)?;
            m.add_row_values(row - group * group_rows, &mut out, &mut word);
        }
        Ok(out)
    }
}
