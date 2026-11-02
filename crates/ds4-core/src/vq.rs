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
//!
//! A `vq-release/1` artifact (the VQ tools) carries the same two payload
//! layouts inside `RQVQ` containers, with two differences that matter:
//!
//! - `DQVQ` v2 is byte-identical to the engine's, so it needs only the
//!   codebook width and the index width as parameters.
//! - `DQV3` is NOT the engine's `DQV3`, although the magic is the same. The
//!   tools' header is 24 bytes (the v2 header, a flags word and four pad
//!   bytes), the codebook is inline at a width the container declares, and the
//!   main stream is `min(nbit, 12)` bits per index rather than always 12
//!   (`/data/VQ_tools/c/gpu/build-vq-gpu.c:82-126`,
//!   `crates/vq-format/src/lib.rs:314-361`). [`VqMatrix::from_payload`] reads
//!   that layout; [`VqMatrix::open`] reads the engine's. Each refuses the
//!   other: the engine calls bytes 20..24 `mnb` and requires 12, the tools
//!   write a zero pad word there.

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
/// The VQ tools' v3 header: the v2 header, a flags word and four pad bytes so
/// the inline codebook lands 8-byte aligned (`build-vq-gpu.c:82-98`).
const V3_TOOLS_HEADER: usize = 24;
/// In the tools' v3 header bit 0 means the 13th-bit plane is present; the
/// codebook element width comes from the container header, not from here
/// (`build-vq-gpu.c:89`).
const V3_TOOLS_FLAG_PLANE: u32 = 1;

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

/// Half to float, subnormals included.
///
/// The reference is the producer's decoder,
/// `/data/VQ_Tools/crates/vq-format/src/lib.rs:86-108`, which starts the
/// normalization exponent at 113 and decrements once per left shift. A
/// transcription of ds4's C copy (`ds41_vq_fmt.h:37-43`, `ds4vq_f16`) had it
/// at 112, which makes every subnormal half decode at half its value: the
/// 2026-11-02 device gate found it on a one-hot probe whose only nonzero
/// codebook entry was 5.1558e-05 and whose CPU dot came out 2.578e-05. The
/// C copy in this tree still carries the off-by-one and is not this
/// family's decoder; it is recorded, not silently shared.
pub fn f16_to_f32(h: u16) -> f32 {
    let s = (u32::from(h) & 0x8000) << 16;
    let e = (u32::from(h) >> 10) & 0x1F;
    let m = u32::from(h) & 0x3FF;
    let f = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e = 113u32;
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

/// One matrix, opened from its payload. Mirrors `v41_vq_mat` and
/// `v41_vq_open` (`cuda_vq_row.inc.cu:19-49`).
#[derive(Debug)]
pub struct VqMatrix<'a> {
    /// Codebook bytes: `nc * dim` entries of [`Self::cb_width`] bytes each.
    codebook: &'a [u8],
    /// Row gains, f16, `rows` entries.
    gains: &'a [u8],
    /// Main index stream.
    stream: &'a [u8],
    /// The 13th-bit plane, when the layer carries one.
    plane: Option<&'a [u8]>,
    v3: bool,
    /// Codebook elements per codeword. The V4.1 recipe is 8
    /// (`cuda_vq_row.inc.cu:31`); the router of a wider model is 4.
    pub dim: u16,
    /// Codebook element width: 2 = f16, 1 = E4M3FN.
    cb_width: usize,
    /// Bits per index in the v3 main stream. 0 for v2, whose indices run
    /// across the whole matrix at [`Self::nbit`] bits each.
    mb: u32,
    /// v3 main-stream and plane bytes per row. 0 for v2.
    mrow: usize,
    prow: usize,
    pub nc: u32,
    pub nbit: u32,
    pub rows: u32,
    pub cols: u32,
}

/// The slices a parsed payload resolves to, before the shape is attached.
struct Fields<'a> {
    codebook: &'a [u8],
    gains: &'a [u8],
    stream: &'a [u8],
    plane: Option<&'a [u8]>,
    v3: bool,
    mb: u32,
    mrow: usize,
    prow: usize,
}

/// Index width from the codebook word count, the same rule as the
/// quantizer's `vq_nbits()` (`vq_fmt.h:71`).
fn index_bits(nc: u32) -> u32 {
    let mut nbit = 0u32;
    while (1u32 << nbit) < nc {
        nbit += 1;
    }
    nbit.max(1)
}

/// The fields every payload header starts with, checked against the caller's
/// expectation, plus the index width its codebook word count implies.
fn head(pay: &[u8], dim: u16, rows: u32, cols: u32) -> Result<(u16, u32, u32), VqError> {
    if pay.len() < V2_HEADER {
        return Err(VqError::Truncated);
    }
    let p_dim = u16_le(pay, 4);
    let nc = u32::from(u16_le(pay, 6));
    if u32_le(pay, 8) != rows || u32_le(pay, 12) != cols {
        return Err(VqError::BadShape);
    }
    if p_dim != dim {
        return Err(VqError::BadDim(p_dim));
    }
    // A codeword is `dim` elements wide, so a row must split into whole
    // codewords; both producers' containers require `cols % dim == 0` and a
    // payload that did not would leave a tail column undefined.
    if p_dim == 0 || nc == 0 || cols % u32::from(p_dim) != 0 {
        return Err(VqError::BadShape);
    }
    Ok((p_dim, nc, index_bits(nc)))
}

/// The 3-byte little-endian window both layouts read an index out of.
///
/// A window at the end of a stream can name bytes the payload does not carry.
/// Those bits lie above the index's own width, so a missing byte reads as
/// zero — the same fallback the tools reader uses
/// (`/data/VQ_tools/crates/vq-format/src/lib.rs:406-411`).
#[inline]
fn window24(s: &[u8], at: usize) -> u32 {
    let byte = |k: usize| s.get(k).copied().unwrap_or(0) as u32;
    byte(at) | (byte(at + 1) << 8) | (byte(at + 2) << 16)
}

impl<'a> VqMatrix<'a> {
    /// Open the matrix for expert `e`, matrix `which`, asserting the payload's
    /// own shape. `rows`/`cols` are the caller's expectation, as in the C
    /// `v41_vq_open`, which returns not-ok rather than trusting the payload.
    ///
    /// This reads the engine's container (`DQVL`) and accepts the V4.1 recipe
    /// only: the payload's `dim` must be [`VQ_DIM`]. A container written by
    /// the VQ tools is read with [`Self::from_payload`].
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
        let (dim, nc, nbit) = head(pay, VQ_DIM, rows, cols)?;
        let nidx_row = cols / u32::from(dim);
        if !v3 {
            // v2 numbers indices across the whole matrix, so the 3-byte window
            // of the last index decides how many stream bytes must exist. The
            // payload carries a safety byte past the packed bits (`vq_fmt.h:5`).
            let total = rows as usize * nidx_row as usize;
            let need = if nbit == 8 {
                total
            } else {
                (((total - 1) * nbit as usize) >> 3) + 3
            };
            return Self::parse_v2(pay, 2, dim, nc, rows, cols, need);
        }
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
        let cb_len = nc as usize * usize::from(dim);
        if cb_off + cb_len > blob.len() {
            return Err(VqError::Truncated);
        }
        let codebook = &blob[cb_off..cb_off + cb_len];
        let tail = &pay[V3_HEADER..];
        if tail.len() < rows as usize * 2 {
            return Err(VqError::Truncated);
        }
        let gains = &tail[..rows as usize * 2];
        let stream = &tail[rows as usize * 2..];
        let mrow = (nidx_row as usize * V3_MAIN_BITS as usize + 7) / 8;
        let main_len = rows as usize * mrow;
        // The last row's 12-bit window may reach two bytes past its packed
        // bits, which the payload covers with a safety byte (`vq_fmt.h:5`).
        let last_window = (rows as usize - 1) * mrow + ((nidx_row as usize - 1) * 12 >> 3) + 3;
        if stream.len() < main_len.max(last_window) {
            return Err(VqError::Truncated);
        }
        let mut prow = 0;
        let mut plane = None;
        if plane_present {
            prow = (nidx_row as usize + 7) / 8;
            let plane_len = rows as usize * prow;
            if stream.len() < main_len + plane_len {
                return Err(VqError::Truncated);
            }
            plane = Some(&stream[main_len..main_len + plane_len]);
        }
        Ok(Self::attach(
            dim,
            1,
            nc,
            rows,
            cols,
            Fields {
                codebook,
                gains,
                stream,
                plane,
                v3: true,
                mb: V3_MAIN_BITS,
                mrow,
                prow,
            },
        ))
    }

    /// Parse one payload of a VQ tools `RQVQ` container, which is what a
    /// `vq-release/1` artifact carries.
    ///
    /// Both payload layouts those containers hold are accepted:
    ///
    /// - `DQVQ` v2: the engine's layout, a 16-byte header, `nc * dim` f16
    ///   codebook entries, gains, and indices packed across the whole matrix
    ///   (`/data/VQ_tools/crates/vq-format/src/lib.rs:343-361`).
    /// - `DQV3`: a 24-byte header (the v2 header, a flags word and four pad
    ///   bytes), an inline codebook of `cb_width` bytes per element, a
    ///   row-local main stream of `min(nbit, 12)` bits and, above 12 bits, a
    ///   one-bit plane (`/data/VQ_tools/c/gpu/build-vq-gpu.c:82-126`).
    ///
    /// The tools' `DQV3` is not the engine's, although the magic matches: the
    /// engine's header is 32 bytes, its codebook sits outside the payload at
    /// `cb_off`, its main stream is always 12 bits and its codebook is always
    /// E4M3. Each constructor refuses the other's header (the engine's `mnb`
    /// is 12 where the tools write a zero pad word), so a format cannot be
    /// decoded under the wrong rules.
    ///
    /// `cb_width` is the container header's word: 2 for f16, 1 for E4M3FN.
    pub fn from_payload(
        pay: &'a [u8],
        cb_width: usize,
        dim: u16,
        rows: u32,
        cols: u32,
    ) -> Result<VqMatrix<'a>, VqError> {
        if cb_width != 1 && cb_width != 2 {
            return Err(VqError::BadShape);
        }
        if pay.len() < V2_HEADER {
            return Err(VqError::Truncated);
        }
        let v3 = match u32_le(pay, 0) {
            MAT_MAGIC => false,
            MAT3_MAGIC => true,
            _ => return Err(VqError::BadPayloadMagic),
        };
        let hdr = if v3 { V3_TOOLS_HEADER } else { V2_HEADER };
        if pay.len() < hdr {
            return Err(VqError::Truncated);
        }
        let (dim, nc, nbit) = head(pay, dim, rows, cols)?;
        let nidx_row = (cols / u32::from(dim)) as usize;
        if !v3 {
            let nidx = rows as usize * nidx_row;
            let idx_bytes = if nbit == 8 {
                nidx
            } else {
                (nidx * nbit as usize + 7) / 8 + 1
            };
            return Self::parse_v2(pay, cb_width, dim, nc, rows, cols, idx_bytes);
        }
        let flags = u32_le(pay, 16);
        // Bytes 20..24 are the writer's zero pad word; a non-zero one is the
        // engine's `mnb` and belongs to the other constructor.
        if u32_le(pay, 20) != 0 || flags & !V3_TOOLS_FLAG_PLANE != 0 {
            return Err(VqError::BadFlags);
        }
        let plane_present = flags & V3_TOOLS_FLAG_PLANE != 0;
        if plane_present != (nbit > V3_MAIN_BITS) {
            return Err(VqError::BadFlags);
        }
        // The writer refuses `nc` above 8192: the main stream plus the plane
        // carries 13 bits, and a wider index would truncate with no error
        // (`build-vq-gpu.c:285-293`).
        if nbit > V3_MAIN_BITS + 1 {
            return Err(VqError::BadFlags);
        }
        let cb_len = nc as usize * usize::from(dim) * cb_width;
        let gain_len = rows as usize * 2;
        if pay.len() < hdr + cb_len + gain_len {
            return Err(VqError::Truncated);
        }
        let mb = nbit.min(V3_MAIN_BITS);
        let mrow = (nidx_row * mb as usize + 7) / 8;
        let prow = if plane_present { (nidx_row + 7) / 8 } else { 0 };
        if pay.len() < hdr + cb_len + gain_len + rows as usize * (mrow + prow) {
            return Err(VqError::Truncated);
        }
        let codebook = &pay[hdr..hdr + cb_len];
        let gains = &pay[hdr + cb_len..hdr + cb_len + gain_len];
        let stream = &pay[hdr + cb_len + gain_len..];
        let plane = if plane_present {
            let at = rows as usize * mrow;
            Some(&stream[at..at + rows as usize * prow])
        } else {
            None
        };
        Ok(Self::attach(
            dim,
            cb_width,
            nc,
            rows,
            cols,
            Fields {
                codebook,
                gains,
                stream,
                plane,
                v3: true,
                mb,
                mrow,
                prow,
            },
        ))
    }

    /// The v2 body: `[16-byte header][nc*dim codebook][rows gains][stream]`.
    /// `idx_bytes` is the producer's own sufficiency rule for the stream.
    fn parse_v2(
        pay: &'a [u8],
        cb_width: usize,
        dim: u16,
        nc: u32,
        rows: u32,
        cols: u32,
        idx_bytes: usize,
    ) -> Result<VqMatrix<'a>, VqError> {
        let cb_len = nc as usize * usize::from(dim) * cb_width;
        let gain_len = rows as usize * 2;
        if pay.len() < V2_HEADER + cb_len + gain_len {
            return Err(VqError::Truncated);
        }
        let body = &pay[V2_HEADER..];
        let codebook = &body[..cb_len];
        let gains = &body[cb_len..cb_len + gain_len];
        let stream = &body[cb_len + gain_len..];
        if stream.len() < idx_bytes {
            return Err(VqError::Truncated);
        }
        Ok(Self::attach(
            dim,
            cb_width,
            nc,
            rows,
            cols,
            Fields {
                codebook,
                gains,
                stream,
                plane: None,
                v3: false,
                mb: 0,
                mrow: 0,
                prow: 0,
            },
        ))
    }

    fn attach(dim: u16, cb_width: usize, nc: u32, rows: u32, cols: u32, f: Fields<'a>) -> Self {
        Self {
            codebook: f.codebook,
            gains: f.gains,
            stream: f.stream,
            plane: f.plane,
            v3: f.v3,
            dim,
            cb_width,
            mb: f.mb,
            mrow: f.mrow,
            prow: f.prow,
            nc,
            nbit: index_bits(nc),
            rows,
            cols,
        }
    }

    /// Index slots in one row: `cols / dim`.
    #[inline]
    fn nidx_row(&self) -> usize {
        self.cols as usize / usize::from(self.dim)
    }

    /// Gain for a row: the payload's f16, as the C decode uses it
    /// (`vq_fmt.h:77`).
    pub fn gain(&self, row: usize) -> f32 {
        f16_to_f32(u16_le(self.gains, row * 2))
    }

    /// Index of the `i`th group of `dim` columns in `row`.
    ///
    /// v2 reads `ceil(log2(nc))` bits little-endian out of a 3-byte window
    /// (`vq_fmt.h:71-80`); its bit positions are numbered across the whole
    /// matrix, not per row. v3 reads the row-local main field of [`Self::mb`]
    /// bits and, when the layer has one, takes its top bit from the plane
    /// (`cuda_vq_row.inc.cu:199-203`).
    pub fn index(&self, row: usize, i: usize) -> u32 {
        let nidx_row = self.nidx_row();
        if !self.v3 {
            let nbit = self.nbit as usize;
            let gi = row * nidx_row + i;
            if nbit == 8 {
                return u32::from(self.stream[gi]);
            }
            let bit = gi * nbit;
            let imsk = if self.nbit >= 32 {
                u32::MAX
            } else {
                (1u32 << self.nbit) - 1
            };
            return (window24(self.stream, bit >> 3) >> (bit & 7)) & imsk;
        }
        let row_stream = &self.stream[row * self.mrow..];
        let bit = i * self.mb as usize;
        let mut v = (window24(row_stream, bit >> 3) >> (bit & 7)) & ((1u32 << self.mb) - 1);
        if let Some(plane) = self.plane {
            let byte = plane[row * self.prow + (i >> 3)];
            v |= u32::from((byte >> (i & 7)) & 1) << self.mb;
        }
        v
    }

    /// One codebook codeword: `dim` f32 values, f16 words or E4M3 bytes.
    pub fn codebook_word_into(&self, idx: usize, out: &mut [f32]) {
        let dim = usize::from(self.dim);
        debug_assert_eq!(out.len(), dim);
        if self.cb_width == 1 {
            for (d, slot) in out.iter_mut().enumerate().take(dim) {
                *slot = e4m3fn_to_f32(self.codebook[idx * dim + d]);
            }
        } else {
            for (d, slot) in out.iter_mut().enumerate().take(dim) {
                *slot = f16_to_f32(u16_le(self.codebook, idx * dim * 2 + d * 2));
            }
        }
    }

    /// The V4.1 `vq8` codeword. Panics if the payload's `dim` is not 8; a
    /// wider or narrower model reads through [`Self::codebook_word_into`].
    pub fn codebook_word(&self, idx: usize) -> [f32; 8] {
        assert_eq!(self.dim, VQ_DIM, "codebook_word is the vq8 accessor");
        let mut c = [0f32; 8];
        self.codebook_word_into(idx, &mut c);
        c
    }

    /// Dequantise the whole matrix row-major, the CPU oracle for
    /// `ds4vq_dequant_f32` (`vq_fmt.h:57-88`). `gain_override` is the engine's
    /// `gov` (a per-row f32 gain from `gr_Lnn.bin`), which replaces the
    /// payload's f16 gain when present (`cuda_vq_row.inc.cu:19,248`).
    pub fn dequant_f32(&self, gain_override: Option<&[f32]>) -> Vec<f32> {
        let mut out = vec![0f32; self.rows as usize * self.cols as usize];
        self.add_rows(&mut out, 0, gain_override);
        out
    }

    /// Add this payload's reconstruction into a row-major tensor at `row0`,
    /// the way a residual stage accumulates into the tensor
    /// (`/data/VQ_tools/crates/vq-format/src/lib.rs:445-461`). `out` must be
    /// zeroed: every value is summed into it.
    pub fn add_into(&self, out: &mut [f32], row0: usize) {
        self.add_rows(out, row0, None);
    }

    /// `x` dotted with one row, decoding the row's codewords on the way.
    ///
    /// This is the shape a forward pass needs: a 360448 x 2816 expert stack
    /// is 4 GB decoded, so the activation is dotted against one row while the
    /// row is still indices. The arithmetic mirrors the tools'
    /// `Stage::dot_row_local` (`/data/VQ_Tools/crates/vq-format/src/lib.rs:468-478`)
    /// exactly: one f32 accumulator, indices in order, each codeword summed in
    /// dim order, and the row gain applied **after** the sum, not folded into
    /// the terms.
    pub fn dot_row(&self, row: usize, x: &[f32]) -> f32 {
        let mut word = vec![0f32; usize::from(self.dim)];
        self.dot_row_scratch(row, x, &mut word)
    }

    /// Add every row's [`Self::dot_row`] into `acc` at `row0`, the accumulator
    /// contract of the tools' `dot_row_local` (`acc[row0 + r] +=`).
    pub fn add_dot_into(&self, x: &[f32], acc: &mut [f32], row0: usize) {
        let mut word = vec![0f32; usize::from(self.dim)];
        for r in 0..self.rows as usize {
            acc[row0 + r] += self.dot_row_scratch(r, x, &mut word);
        }
    }

    /// [`Self::dot_row`] with the caller's codeword buffer, so a matrix loop
    /// allocates once. A forward that gathers one row of a fused expert stack
    /// per selected expert calls this directly.
    pub fn dot_row_scratch(&self, row: usize, x: &[f32], word: &mut [f32]) -> f32 {        let dim = usize::from(self.dim);
        let nidx_row = self.nidx_row();
        let mut s = 0.0f32;
        for i in 0..nidx_row {
            self.codebook_word_into(self.index(row, i) as usize, word);
            for d in 0..dim {
                s += word[d] * x[i * dim + d];
            }
        }
        s * self.gain(row)
    }

    /// Add one row's dequantised values, gain included, into `out`, whose
    /// length is the payload's `cols`. This is the single-row form of
    /// [`Self::add_into`] a token-embedding lookup needs: the table is
    /// 262144 rows and only one is read.
    pub fn add_row_values(&self, row: usize, out: &mut [f32], word: &mut [f32]) {        let dim = usize::from(self.dim);
        let nidx_row = self.nidx_row();
        debug_assert_eq!(out.len(), self.cols as usize);
        let gain = self.gain(row);
        for i in 0..nidx_row {
            self.codebook_word_into(self.index(row, i) as usize, word);
            let o = &mut out[i * dim..][..dim];
            for d in 0..dim {
                o[d] += word[d] * gain;
            }
        }
    }

    /// The whole codebook as f32, in codeword order. A row walk over a large
    /// matrix converts each codeword once instead of once per use.
    pub fn codebook_f32(&self) -> Vec<f32> {
        let dim = usize::from(self.dim);
        let mut out = vec![0f32; self.nc as usize * dim];
        for (idx, chunk) in out.chunks_exact_mut(dim).enumerate() {
            self.codebook_word_into(idx, chunk);
        }
        out
    }

    /// [`Self::dot_row`] with the codebook already decoded to f32 by
    /// [`Self::codebook_f32`]: the same arithmetic, without a per-index
    /// f16-to-f32 conversion or a copy into a scratch word.
    pub fn dot_row_codebook(&self, row: usize, x: &[f32], codebook: &[f32]) -> f32 {
        let dim = usize::from(self.dim);
        let nidx_row = self.nidx_row();
        let mut s = 0.0f32;
        for i in 0..nidx_row {
            let idx = self.index(row, i) as usize;
            let word = &codebook[idx * dim..idx * dim + dim];
            let xw = &x[i * dim..i * dim + dim];
            for d in 0..dim {
                s += word[d] * xw[d];
            }
        }
        s * self.gain(row)
    }

    fn add_rows(&self, out: &mut [f32], row0: usize, gain_override: Option<&[f32]>) {
        let dim = usize::from(self.dim);
        let nidx_row = self.nidx_row();
        let cols = self.cols as usize;
        let mut word = vec![0f32; dim];
        for r in 0..self.rows as usize {
            let gain = match gain_override {
                Some(g) => g[r],
                None => self.gain(r),
            };
            let base = (row0 + r) * cols;
            for i in 0..nidx_row {
                self.codebook_word_into(self.index(r, i) as usize, &mut word);
                let o = &mut out[base + i * dim..][..dim];
                for d in 0..dim {
                    o[d] += word[d] * gain;
                }
            }
        }
    }
}
