//! DeepSeek V4.1 VQ expert-blob decode: `DQVL` container, `DQVQ` v2 and
//! `DQV3` v3 payloads.
//!
//! Source of truth is the C engine at `/data/YoungAi` (commit 3946dbc), and
//! every field below cites it:
//!
//! - `vq_fmt.h:1-7`   container layout and the value rule
//! - `vq_fmt.h:15-20` magics and the accepted version range (2..=3)
//! - `vq_fmt.h:31-53` header accessors and the slot table at offset 16
//! - `vq_fmt.h:57-88` v2 decode: per-payload f16 codebook, f16 gain per row,
//!   index width `ceil(log2(nc))` with a byte stream at 8 bits and a LE bit
//!   window otherwise
//! - `src/cuda/cuda_vq_row.inc.cu:19-49` v3 geometry: one E4M3 codebook per
//!   layer at `cb_off`, a fixed 12-bit main stream, and for 13-bit layers a
//!   1-bit plane whose bit j is the 13th bit of index j
//! - value rule (`vq_fmt.h:7`): `value[row][col] = codebook[idx][col % 8] * gain[row]`
//!
//! This is the CPU oracle. The device path reads the same bytes with warp
//! block loads (`cuda_vq_row.inc.cu:184-249`); its output must equal this
//! decoder's on the same payload.

use std::fmt;

/// `DS4VQ_BLOB_MAGIC` 'DQVL' (`vq_fmt.h:15`).
pub const BLOB_MAGIC: u32 = 0x4C56_5144;
/// `DS4VQ_MAT_MAGIC` 'DQVQ', the v2 payload (`vq_fmt.h:16`).
pub const MAT_MAGIC: u32 = 0x5156_5144;
/// `DS4VQ_MAT3_MAGIC` 'DQV3', the v3 payload (`vq_fmt.h:19`).
pub const MAT3_MAGIC: u32 = 0x3356_5144;
/// Accepted container versions, inclusive (`vq_fmt.h:19-20`).
pub const BLOB_VER_MIN: u32 = 2;
pub const BLOB_VER_MAX: u32 = 3;
/// The v3 main stream is always 12 bits; the 13th lives in the plane
/// (`cuda_vq_row.inc.cu:34`).
pub const V3_MAIN_BITS: u32 = 12;
/// v3 payload flags: bit 0 = the codebook is E4M3, bit 1 = a 13th-bit plane
/// is present (`cuda_vq_row.inc.cu:35,41`).
pub const V3_FLAG_E4M3: u32 = 1;
pub const V3_FLAG_PLANE: u32 = 2;
/// The V4.1 recipe is vq8; the device open rejects any other dim
/// (`cuda_vq_row.inc.cu:31`).
pub const VQ_DIM: u16 = 8;
/// Payload header bytes: magic, dim, nc, rows, cols (`vq_fmt.h:57`).
const V2_HEADER: usize = 16;
/// v3 payload header bytes: the v2 header plus flags, mnb and cb_off
/// (`cuda_vq_row.inc.cu:33-39`).
const V3_HEADER: usize = 32;

#[derive(Debug, PartialEq, Eq)]
pub enum VqError {
    TooShort,
    BadBlobMagic,
    BadVersion(u32),
    BadNexp(u32),
    NoMatrix,
    BadPayloadMagic,
    BadShape,
    BadDim(u16),
    BadFlags,
    Truncated,
}

impl VqError {
    pub fn token(&self) -> String {
        match self {
            VqError::TooShort => "vq-too-short".into(),
            VqError::BadBlobMagic => "vq-blob-magic".into(),
            VqError::BadVersion(v) => format!("vq-blob-version {v}"),
            VqError::BadNexp(n) => format!("vq-blob-nexp {n}"),
            VqError::NoMatrix => "vq-no-matrix".into(),
            VqError::BadPayloadMagic => "vq-payload-magic".into(),
            VqError::BadShape => "vq-payload-shape".into(),
            VqError::BadDim(d) => format!("vq-payload-dim {d}"),
            VqError::BadFlags => "vq-payload-flags".into(),
            VqError::Truncated => "vq-payload-truncated".into(),
        }
    }
}

impl fmt::Display for VqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for VqError {}

fn u16_le(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn u32_le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn u64_le(b: &[u8], off: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(w)
}

/// C `ds4vq_f16` (`vq_fmt.h:22-28`): a half decode that keeps subnormals
/// exact, including the negative-zero and NaN encodings.
pub fn f16_to_f32(h: u16) -> f32 {
    let s = (u32::from(h) & 0x8000) << 16;
    let e = (u32::from(h) >> 10) & 0x1F;
    let m = u32::from(h) & 0x3FF;
    let f = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e = 112u32;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if e == 31 {
        s | 0x7F80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(f)
}

/// C `ds4_e4m3fn_to_f32` (`src/common/ds4_fp8.h:29-38`), the single E4M3
/// primitive the engine uses: 0x7f/0xff are NaN (E4M3FN has no Inf), and
/// exp 0 is the subnormal `mant * 2^-9`.
pub fn e4m3fn_to_f32(x: u8) -> f32 {
    let abs = x & 0x7F;
    let sign = x & 0x80 != 0;
    if abs == 0 {
        return if sign { -0.0 } else { 0.0 };
    }
    if abs == 0x7F {
        return f32::NAN;
    }
    let exp = i32::from((x >> 3) & 0x0F);
    let man = i32::from(x & 0x07);
    let value = if exp == 0 {
        (man as f32) * 2f32.powi(-9)
    } else {
        (1.0 + (man as f32) / 8.0) * 2f32.powi(exp - 7)
    };
    if sign {
        -value
    } else {
        value
    }
}

/// C `ds4vq_blob_ver` (`vq_fmt.h:34-37`). Panics on a short slice; callers
/// gate on [`blob_ok`] first.
pub fn blob_ver(blob: &[u8]) -> u32 {
    u32_le(blob, 4)
}

/// C `ds4vq_blob_nexp` (`vq_fmt.h:31-33`).
pub fn blob_nexp(blob: &[u8]) -> u32 {
    u32_le(blob, 12)
}

/// C `ds4vq_blob_ok` (`vq_fmt.h:43-49`): magic, version whitelist, and a slot
/// table that fits. An unknown version is a hard load-time stop in the engine,
/// because decoding a future layout as this one yields plausible wrong weights
/// rather than a diagnosable failure.
pub fn blob_ok(blob: &[u8]) -> bool {
    if blob.len() < 16 {
        return false;
    }
    if u32_le(blob, 0) != BLOB_MAGIC {
        return false;
    }
    let ver = blob_ver(blob);
    if !(BLOB_VER_MIN..=BLOB_VER_MAX).contains(&ver) {
        return false;
    }
    let nexp = blob_nexp(blob);
    (1..=4096).contains(&nexp) && blob.len() >= 16 + nexp as usize * 3 * 8
}

/// C `ds4vq_slot` (`vq_fmt.h:51-54`): byte offset of expert `e`'s matrix
/// `which` (0 = w1, 1 = w3, 2 = w2), or 0 when absent. The slot table sits at
/// offset 16 in both container versions.
pub fn blob_slot(blob: &[u8], e: usize, which: usize) -> u64 {
    u64_le(blob, 16 + (e * 3 + which) * 8)
}

/// One expert matrix, opened from its payload. Mirrors `v41_vq_mat` and
/// `v41_vq_open` (`cuda_vq_row.inc.cu:19-49`).
#[derive(Debug)]
pub struct VqMatrix<'a> {
    /// Codebook bytes: `nc * 8` f16 entries in v2, E4M3 bytes in v3.
    codebook: &'a [u8],
    /// Row gains, f16, `rows` entries.
    gains: &'a [u8],
    /// Main index stream.
    stream: &'a [u8],
    /// The 13th-bit plane, when the layer carries one.
    plane: Option<&'a [u8]>,
    v3: bool,
    pub nc: u32,
    pub nbit: u32,
    pub rows: u32,
    pub cols: u32,
}

impl<'a> VqMatrix<'a> {
    /// Open the matrix for expert `e`, matrix `which`, asserting the payload's
    /// own shape. `rows`/`cols` are the caller's expectation, as in the C
    /// `v41_vq_open`, which returns not-ok rather than trusting the payload.
    pub fn open(
        blob: &'a [u8],
        e: usize,
        which: usize,
        rows: u32,
        cols: u32,
    ) -> Result<VqMatrix<'a>, VqError> {
        if !blob_ok(blob) {
            return Err(VqError::BadBlobMagic);
        }
        let off = blob_slot(blob, e, which);
        if off == 0 {
            return Err(VqError::NoMatrix);
        }
        let off = off as usize;
        if off + V3_HEADER > blob.len() {
            return Err(VqError::Truncated);
        }
        let pay = &blob[off..];
        let magic = u32_le(pay, 0);
        let v3 = match magic {
            MAT_MAGIC => false,
            MAT3_MAGIC => true,
            _ => return Err(VqError::BadPayloadMagic),
        };
        let dim = u16_le(pay, 4);
        let nc = u32::from(u16_le(pay, 6));
        let p_rows = u32_le(pay, 8);
        let p_cols = u32_le(pay, 12);
        if p_rows != rows || p_cols != cols {
            return Err(VqError::BadShape);
        }
        if dim != VQ_DIM {
            return Err(VqError::BadDim(dim));
        }
        // Index width from the codebook word count, the same rule as the
        // quantizer's `vq_nbits()` (`vq_fmt.h:71`).
        let mut nbit = 0u32;
        while (1u32 << nbit) < nc {
            nbit += 1;
        }
        if nbit < 1 {
            nbit = 1;
        }
        let nidx_row = cols / u32::from(VQ_DIM);
        let mut out = VqMatrix {
            codebook: &[],
            gains: &[],
            stream: &[],
            plane: None,
            v3,
            nc,
            nbit,
            rows,
            cols,
        };
        if v3 {
            let flags = u32_le(pay, 16);
            let mnb = u32_le(pay, 20);
            let cb_off = u64_le(pay, 24) as usize;
            if mnb != V3_MAIN_BITS || flags & V3_FLAG_E4M3 == 0 {
                return Err(VqError::BadFlags);
            }
            let plane_present = flags & V3_FLAG_PLANE != 0;
            if plane_present != (nbit > V3_MAIN_BITS) {
                return Err(VqError::BadFlags);
            }
            let cb_len = nc as usize * usize::from(VQ_DIM);
            if cb_off + cb_len > blob.len() {
                return Err(VqError::Truncated);
            }
            out.codebook = &blob[cb_off..cb_off + cb_len];
            let gains = &pay[V3_HEADER..];
            if gains.len() < rows as usize * 2 {
                return Err(VqError::Truncated);
            }
            out.gains = &gains[..rows as usize * 2];
            let stream = &gains[rows as usize * 2..];
            let mrow = (nidx_row as usize * V3_MAIN_BITS as usize + 7) / 8;
            let main_len = rows as usize * mrow;
            // The last row's 12-bit window may reach two bytes past its packed
            // bits, which the payload covers with a safety byte (`vq_fmt.h:5`).
            let last_window = (rows as usize - 1) * mrow + ((nidx_row as usize - 1) * 12 >> 3) + 3;
            if stream.len() < main_len.max(last_window) {
                return Err(VqError::Truncated);
            }
            out.stream = stream;
            if plane_present {
                let prow = (nidx_row as usize + 7) / 8;
                let plane_len = rows as usize * prow;
                if stream.len() < main_len + plane_len {
                    return Err(VqError::Truncated);
                }
                out.plane = Some(&stream[main_len..main_len + plane_len]);
            }
        } else {
            let cb_len = nc as usize * usize::from(VQ_DIM) * 2;
            let gains = &pay[V2_HEADER..];
            if gains.len() < cb_len + rows as usize * 2 {
                return Err(VqError::Truncated);
            }
            out.codebook = &gains[..cb_len];
            out.gains = &gains[cb_len..cb_len + rows as usize * 2];
            out.stream = &gains[cb_len + rows as usize * 2..];
            // v2 numbers indices across the whole matrix, so the 3-byte window
            // of the last index decides how many stream bytes must exist. The
            // payload carries a safety byte past the packed bits (`vq_fmt.h:5`).
            let total = rows as usize * nidx_row as usize;
            let need = if nbit == 8 {
                total
            } else {
                (((total - 1) * nbit as usize) >> 3) + 3
            };
            if out.stream.len() < need {
                return Err(VqError::Truncated);
            }
        }
        Ok(out)
    }

    /// Gain for a row: the payload's f16, as the C decode uses it
    /// (`vq_fmt.h:77`).
    pub fn gain(&self, row: usize) -> f32 {
        f16_to_f32(u16_le(self.gains, row * 2))
    }

    /// Index of the `i`th group of 8 columns in `row`.
    ///
    /// v2 reads `ceil(log2(nc))` bits little-endian out of a 3-byte window
    /// (`vq_fmt.h:71-80`); its bit positions are numbered across the whole
    /// matrix, not per row. v3 reads a fixed 12-bit main field from the row's
    /// own span and takes the 13th bit from the plane
    /// (`cuda_vq_row.inc.cu:199-203`).
    pub fn index(&self, row: usize, i: usize) -> u32 {
        let nidx_row = (self.cols / u32::from(VQ_DIM)) as usize;
        if !self.v3 {
            let nbit = self.nbit as usize;
            let gi = row * nidx_row + i;
            if nbit == 8 {
                return u32::from(self.stream[gi]);
            }
            let bit = gi * nbit;
            let w = u32::from(self.stream[bit >> 3])
                | u32::from(self.stream[(bit >> 3) + 1]) << 8
                | u32::from(self.stream[(bit >> 3) + 2]) << 16;
            let imsk = if self.nbit >= 32 {
                u32::MAX
            } else {
                (1u32 << self.nbit) - 1
            };
            return (w >> (bit & 7)) & imsk;
        }
        let mrow = (nidx_row * V3_MAIN_BITS as usize + 7) / 8;
        let row_stream = &self.stream[row * mrow..];
        let bit = i * V3_MAIN_BITS as usize;
        let w = u32::from(row_stream[bit >> 3])
            | u32::from(row_stream[(bit >> 3) + 1]) << 8
            | u32::from(row_stream[(bit >> 3) + 2]) << 16;
        let mut v = (w >> (bit & 7)) & ((1u32 << V3_MAIN_BITS) - 1);
        if let Some(plane) = self.plane {
            let prow = (nidx_row + 7) / 8;
            let byte = plane[row * prow + (i >> 3)];
            v |= u32::from((byte >> (i & 7)) & 1) << V3_MAIN_BITS;
        }
        v
    }

    /// The `nc * 8` codebook entries as f32: f16 words in v2, E4M3 bytes in v3.
    pub fn codebook_word(&self, idx: usize) -> [f32; 8] {
        let mut c = [0f32; 8];
        if self.v3 {
            for (d, slot) in c.iter_mut().enumerate() {
                *slot = e4m3fn_to_f32(self.codebook[idx * 8 + d]);
            }
        } else {
            for (d, slot) in c.iter_mut().enumerate() {
                *slot = f16_to_f32(u16_le(self.codebook, idx * 16 + d * 2));
            }
        }
        c
    }

    /// Dequantise the whole matrix row-major, the CPU oracle for
    /// `ds4vq_dequant_f32` (`vq_fmt.h:57-88`). `gain_override` is the engine's
    /// `gov` (a per-row f32 gain from `gr_Lnn.bin`), which replaces the
    /// payload's f16 gain when present (`cuda_vq_row.inc.cu:19,248`).
    pub fn dequant_f32(&self, gain_override: Option<&[f32]>) -> Vec<f32> {
        let dim = usize::from(VQ_DIM);
        let nidx_row = (self.cols / u32::from(VQ_DIM)) as usize;
        let cols = self.cols as usize;
        let mut out = vec![0f32; self.rows as usize * cols];
        for r in 0..self.rows as usize {
            let gain = match gain_override {
                Some(g) => g[r],
                None => self.gain(r),
            };
            for i in 0..nidx_row {
                let c = self.codebook_word(self.index(r, i) as usize);
                let o = &mut out[r * cols + i * dim..][..dim];
                for d in 0..dim {
                    o[d] = c[d] * gain;
                }
            }
        }
        out
    }
}
