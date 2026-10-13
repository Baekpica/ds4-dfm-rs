//! DeepSeek V4.1 sidecars: the `gr`/`rb`/`amp` file formats and the
//! `base.fnv` admission fingerprint.
//!
//! Source of truth is the C engine at `/data/YoungAi` (commit 3946dbc):
//!
//! - `core_v41_amp.c:27-33` `amp_read_mat`: type 43 = fp4x32 blocks, else f32.
//! - `core_v41_amp.c:44-84` `v41_gr_accum_layer`: header `<i32 n_expert><i32 D>
//!   <i32 type>`; type 1 = f32 s, 2 = f16 s, 43 = fp4x32 storing s-1, and the
//!   loader restores `1 + raw`. A present-but-broken file is a hard stop, not a
//!   skip: skipping half a plugin silently produces half a fix.
//! - `core_v41_amp.c:86-117` `v41_rb_load_layer`: header `<i32 n_expert>
//!   <i32 1=f32>` plus the f32 bias.
//! - `core_v41_amp.c:194-223` `amp_read_layer`: header `<i32 D><i32 K><i32
//!   type>`, D must be the embedding width, 0 < K <= 8192, type 1 or 43, then
//!   A[K][D] and B[K][D].
//! - `core_v41_amp.c:159-186` `v41_pt_base_ok`: the post-train directory must
//!   carry a `base.fnv` naming the ② directory it was solved against; a
//!   missing file warns and passes, a mismatch stops.
//! - `ds4_gr_fnv.h:18-33` `ds4_gr_dir_fnv`: FNV-1a 64 over
//!   `gr_L%02u.bin` then `rb_L%02u.bin` per layer, seed 1469598103934665603,
//!   prime 1099511628211, counting the files it found.
//! - `ds4_quantfmt.c:28-40` `ds4_deq_fp4x32` and `ds4_fp8.h:39-44,121-126`:
//!   the E8M0 scale and the E2M1 nibble table.

use std::fs;
use std::io::Read;
use std::path::Path;

use crate::vq::f16_to_f32;

/// `DS4_GR_FNV_SEED` / `DS4_GR_FNV_PRIME` (`ds4_gr_fnv.h:14-15`).
pub const FNV_SEED: u64 = 1469598103934665603;
pub const FNV_PRIME: u64 = 1099511628211;
/// GGUF type ids the sidecars use (`ds4_quantfmt.h:20-45`).
pub const TYPE_F32: i32 = 1;
pub const TYPE_F16: i32 = 2;
pub const TYPE_FP4X32: i32 = 43;
/// `amp_read_layer`'s rank bound (`core_v41_amp.c:199`).
pub const AMP_MAX_RANK: u32 = 8192;

#[derive(Debug)]
pub enum SidecarError {
    Io(String),
    Header(String),
    Type(String),
    Truncated(String),
    /// `base.fnv` names a different ② directory than the one attached.
    BaseMismatch {
        want: u64,
        want_files: u32,
        have: u64,
        have_files: u32,
    },
    /// Neither directory holds any amp, gr or rb file (`core_v41_amp.c:259-263`).
    Empty,
    /// The ② + ③ rank sum exceeds `AMP_MAX_RANK` (`core_v41_amp.c:242`).
    RankOver {
        layer: u32,
        k2: u32,
        k3: u32,
    },
}

impl SidecarError {
    pub fn token(&self) -> String {
        match self {
            SidecarError::Io(p) => format!("sidecar-io {p}"),
            SidecarError::Header(p) => format!("sidecar-header {p}"),
            SidecarError::Type(p) => format!("sidecar-type {p}"),
            SidecarError::Truncated(p) => format!("sidecar-truncated {p}"),
            SidecarError::BaseMismatch { .. } => "sidecar-base-mismatch".into(),
            SidecarError::Empty => "zchain-empty".into(),
            SidecarError::RankOver { layer, k2, k3 } => {
                format!("zchain-rank-over L{layer:02} {k2}+{k3}")
            }
        }
    }
}

impl std::fmt::Display for SidecarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.token())
    }
}

impl std::error::Error for SidecarError {}

/// `ds4_e2m1fn` decode table (`ds4_fp8.h:121-126`).
pub fn fp4_nibble_to_f32(n: u8) -> f32 {
    const T: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    T[(n & 15) as usize]
}

/// `ds4_e8m0_to_f32` (`ds4_fp8.h:39-44`): e == 0 is the 0x00400000 bit pattern
/// (2^-127 subnormal), everything else is 2^(e-127).
pub fn e8m0_to_f32(e: u8) -> f32 {
    let bits: u32 = if e == 0 {
        0x0040_0000
    } else {
        u32::from(e) << 23
    };
    f32::from_bits(bits)
}

/// C `ds4_deq_fp4x32` (`ds4_quantfmt.c:28-40`): 17-byte blocks, byte j low
/// nibble = element 2j, high nibble = element 2j+1, byte 16 the E8M0 scale.
pub fn deq_fp4x32(src: &[u8]) -> Vec<f32> {
    let nblk = src.len() / 17;
    let mut out = Vec::with_capacity(nblk * 32);
    for b in 0..nblk {
        let blk = &src[b * 17..b * 17 + 17];
        let s = e8m0_to_f32(blk[16]);
        for j in 0..16 {
            out.push(fp4_nibble_to_f32(blk[j] & 0x0F) * s);
            out.push(fp4_nibble_to_f32(blk[j] >> 4) * s);
        }
    }
    out
}

fn read_i32_header(bytes: &[u8], n: usize, path: &Path) -> Result<Vec<i32>, SidecarError> {
    if bytes.len() < n * 4 {
        return Err(SidecarError::Truncated(path.display().to_string()));
    }
    Ok(bytes[..n * 4]
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// `gr_Lnn.bin`: per-expert per-channel gain factors.
pub struct GrSidecar {
    pub n_expert: u32,
    pub d: u32,
    pub typ: i32,
    /// The factor each element multiplies the base gain by. For the fp4x32
    /// form the disk stores `s - 1` and this restores `1 + raw`
    /// (`core_v41_amp.c:71-74`).
    pub factor: Vec<f32>,
}

impl GrSidecar {
    /// `v41_gr_accum_layer` (`core_v41_amp.c:44-84`). `None` = no file.
    pub fn read(path: &Path, n_expert: u32, n_embd: u32) -> Result<Option<Self>, SidecarError> {
        let Ok(bytes) = fs::read(path) else {
            return Ok(None);
        };
        let hd = read_i32_header(&bytes, 3, path)?;
        if hd[0] != n_expert as i32
            || hd[1] != n_embd as i32
            || (hd[2] != TYPE_F32 && hd[2] != TYPE_F16 && hd[2] != TYPE_FP4X32)
        {
            return Err(SidecarError::Header(format!(
                "{} expert={} d={} type={}",
                path.display(),
                hd[0],
                hd[1],
                hd[2]
            )));
        }
        let nel = n_expert as usize * n_embd as usize;
        let body = &bytes[12..];
        let factor = match hd[2] {
            TYPE_FP4X32 => {
                if nel % 32 != 0 {
                    return Err(SidecarError::Type(path.display().to_string()));
                }
                let nby = nel / 32 * 17;
                if body.len() < nby {
                    return Err(SidecarError::Truncated(path.display().to_string()));
                }
                deq_fp4x32(&body[..nby])
                    .into_iter()
                    .map(|v| 1.0 + v)
                    .collect()
            }
            TYPE_F16 => {
                if body.len() < nel * 2 {
                    return Err(SidecarError::Truncated(path.display().to_string()));
                }
                body[..nel * 2]
                    .chunks_exact(2)
                    .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
                    .collect()
            }
            _ => {
                if body.len() < nel * 4 {
                    return Err(SidecarError::Truncated(path.display().to_string()));
                }
                body[..nel * 4]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            }
        };
        Ok(Some(Self {
            n_expert,
            d: n_embd,
            typ: hd[2],
            factor,
        }))
    }
}

/// `rb_Lnn.bin`: the router-bias delta added to the selection score.
pub struct RbSidecar {
    pub bias: Vec<f32>,
}

impl RbSidecar {
    /// `v41_rb_load_layer` (`core_v41_amp.c:86-117`). `None` = no file.
    pub fn read(path: &Path, n_expert: u32) -> Result<Option<Self>, SidecarError> {
        let Ok(bytes) = fs::read(path) else {
            return Ok(None);
        };
        let hd = read_i32_header(&bytes, 2, path)?;
        if hd[0] != n_expert as i32 || hd[1] != 1 {
            return Err(SidecarError::Header(format!(
                "{} expert={} type={}",
                path.display(),
                hd[0],
                hd[1]
            )));
        }
        let body = &bytes[8..];
        let n = n_expert as usize;
        if body.len() < n * 4 {
            return Err(SidecarError::Truncated(path.display().to_string()));
        }
        Ok(Some(Self {
            bias: body[..n * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        }))
    }
}

/// `amp_Lnn.bin`: the low-rank correction `y += x * (B * A)`.
pub struct AmpSidecar {
    pub d: u32,
    pub k: u32,
    pub typ: i32,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl AmpSidecar {
    /// `amp_read_layer` (`core_v41_amp.c:194-223`). `None` = no file.
    pub fn read(path: &Path, n_embd: u32) -> Result<Option<Self>, SidecarError> {
        let Ok(bytes) = fs::read(path) else {
            return Ok(None);
        };
        let hd = read_i32_header(&bytes, 3, path)?;
        if hd[0] != n_embd as i32 || hd[1] <= 0 || hd[1] > AMP_MAX_RANK as i32 {
            return Err(SidecarError::Header(format!(
                "{} d={} k={}",
                path.display(),
                hd[0],
                hd[1]
            )));
        }
        if hd[2] != TYPE_F32 && hd[2] != TYPE_FP4X32 {
            return Err(SidecarError::Type(path.display().to_string()));
        }
        let d = hd[0] as u32;
        let k = hd[1] as u32;
        let nel = k as usize * d as usize;
        let mat_bytes = if hd[2] == TYPE_FP4X32 {
            if nel % 32 != 0 {
                return Err(SidecarError::Type(path.display().to_string()));
            }
            nel / 32 * 17
        } else {
            nel * 4
        };
        let body = &bytes[12..];
        if body.len() < mat_bytes * 2 {
            return Err(SidecarError::Truncated(path.display().to_string()));
        }
        let decode = |raw: &[u8]| -> Vec<f32> {
            if hd[2] == TYPE_FP4X32 {
                deq_fp4x32(raw)
            } else {
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            }
        };
        Ok(Some(Self {
            d,
            k,
            typ: hd[2],
            a: decode(&body[..mat_bytes]),
            b: decode(&body[mat_bytes..mat_bytes * 2]),
        }))
    }
}

/// C `ds4_gr_dir_fnv` (`ds4_gr_fnv.h:18-33`): FNV-1a 64 over the ② directory's
/// `gr_L%02u.bin` then `rb_L%02u.bin` per layer, missing layers skipped, with
/// the file count. Both kinds count: either one changing means ③ must be
/// re-solved (`ds4_gr_fnv.h:24`).
pub fn gr_dir_fnv(dir: &Path, n_layer: u32) -> (u64, u32) {
    let mut h = FNV_SEED;
    let mut files = 0u32;
    let mut buf = vec![0u8; 65536];
    for il in 0..n_layer {
        for kind in ["gr", "rb"] {
            let p = dir.join(format!("{kind}_L{il:02}.bin"));
            let Ok(mut f) = fs::File::open(&p) else {
                continue;
            };
            files += 1;
            loop {
                let got = f.read(&mut buf).unwrap_or(0);
                if got == 0 {
                    break;
                }
                for &b in &buf[..got] {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(FNV_PRIME);
                }
            }
        }
    }
    (h, files)
}

/// The result of the `base.fnv` gate (`core_v41_amp.c:159-186`).
#[derive(Debug, PartialEq, Eq)]
pub enum BaseFingerprint {
    /// No `base.fnv`: an experimental post-train build; the engine warns and
    /// passes, and so does this.
    Absent,
    Checked {
        hash: u64,
        files: u32,
    },
}

/// C `v41_pt_base_ok`: recompute the ② directory's fingerprint and compare it
/// with what the ③ directory recorded. A mismatch is a hard stop - the
/// post-train corrections would be applied to the wrong baseline and still
/// produce plausible-looking output (`ds4_gr_fnv.h:3-7`).
pub fn check_base_fnv(
    pt_dir: &Path,
    amp_dir: Option<&Path>,
    n_layer: u32,
) -> Result<BaseFingerprint, SidecarError> {
    let path = pt_dir.join("base.fnv");
    let Ok(text) = fs::read_to_string(&path) else {
        return Ok(BaseFingerprint::Absent);
    };
    let mut it = text.split_whitespace();
    let want = it
        .next()
        .and_then(|v| u64::from_str_radix(v, 16).ok())
        .ok_or_else(|| SidecarError::Header(path.display().to_string()))?;
    let want_files: u32 = it
        .next()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| SidecarError::Header(path.display().to_string()))?;
    let (have, have_files) = match amp_dir {
        Some(dir) => gr_dir_fnv(dir, n_layer),
        None => (FNV_SEED, 0),
    };
    if have != want || have_files != want_files {
        return Err(SidecarError::BaseMismatch {
            want,
            want_files,
            have,
            have_files,
        });
    }
    Ok(BaseFingerprint::Checked {
        hash: have,
        files: have_files,
    })
}
