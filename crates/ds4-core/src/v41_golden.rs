//! The ds41 vision golden gate: 1:1 port of the engine's
//! `src/core/core_v41_vision_gate.c` (`ds4_engine_v41_vision_golden`).
//!
//! The fixture is produced by the engine's own generator
//! (`tools/vision_golden.py`: official `inference/{image_processor,vision}.py`
//! plus the BF16 originals; the generator stays outside this repo, rule 14)
//! and the format is the engine's:
//!   "VGLD" u32 ver=1 | u32 w h n_vit_h n_vit_w n_llm_h n_llm_w pp E
//!   best_w best_h | u8 rgb[best_h][best_w][3] (after PIL pad)
//!   | f32 patches[n_vit][pp] | f32 embeds[n_llm][E]          (gate :3-5)
//!
//! Checks, each naming which segment is sick (gate :6-10):
//!   (1) the size account: the Rust plan must match the golden grid;
//!   (2) pixels: REPORT ONLY (the engine reads the section and discards it,
//!       `(void)grgb`, gate :76; kept here because the port decodes with the
//!       `image` crate, not stb, so the decoder difference is the one signal
//!       this section carries -- a named port addition that never gates);
//!   (3) patches: the production preprocessing vs the golden patches, fail
//!       over 4 pixel-levels (one pixel-level is 2/255, gate :71-74);
//!   (4) the tower: the GOLDEN patches through our tower vs the golden
//!       embeds (this check measures the tower alone), fail under mean row
//!       cosine 0.999 or min row cosine 0.95 (gate :39-43);
//!   (5) end-to-end: our patches through our tower, report only (gate :87-90).

use crate::v41_media::{self, V41VisionMeta};
use crate::{Error, Model, Result};
use ds4_sys::{
    ds4_bridge_v41_has_vision, ds4_bridge_v41_vision_rows_free, ds4_bridge_v41_vision_run,
};
use std::path::Path;

/// gate :42-43.  Basis (gate :39-41): the official bf16 run against the same
/// code in f32 has min row cosine 0.926 / mean 0.99927 on img.jpeg, so the
/// gate judges mean >= 0.999 AND min row >= 0.95; min alone would fail the
/// official run against itself.
pub const VIS_GATE_MEAN_COS: f64 = 0.999;
pub const VIS_GATE_MIN_COS: f64 = 0.95;
/// gate :73: a patch difference of 4 pixel-levels (each level is 2/255).
const VIS_GATE_PATCH_PIXELS: f64 = 4.0 * 2.0 / 255.0;

const VGLD_MAGIC: &[u8; 4] = b"VGLD";
const VGLD_HEADER_BYTES: usize = 4 + 12 * 4;

fn gate_error(e: impl std::fmt::Display) -> Error {
    Error {
        code: 1,
        message: e.to_string(),
    }
}

/// A parsed VGLD fixture (gate :47-56).
pub struct V41Golden {
    pub w: u32,
    pub h: u32,
    pub n_vit_h: u32,
    pub n_vit_w: u32,
    pub n_llm_h: u32,
    pub n_llm_w: u32,
    pub pp: u32,
    pub e: u32,
    pub best_w: u32,
    pub best_h: u32,
    pub rgb: Vec<u8>,      // [best_h][best_w][3], after PIL pad
    pub patches: Vec<f32>, // [n_vit][pp]
    pub embeds: Vec<f32>,  // [n_llm][e]
}

impl V41Golden {
    pub fn n_vit(&self) -> u64 {
        u64::from(self.n_vit_h) * u64::from(self.n_vit_w)
    }

    pub fn n_llm(&self) -> u64 {
        u64::from(self.n_llm_h) * u64::from(self.n_llm_w)
    }

    /// Read and validate a VGLD file.  Bad magic/version or a short read
    /// refuse by name, the two failure shapes the engine has (gate :50-52).
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| gate_error(format!("vision golden {}: {e}", path.display())))?;
        let bad = || {
            gate_error(format!(
                "vision golden {}: bad magic or version",
                path.display()
            ))
        };
        if bytes.len() < VGLD_HEADER_BYTES || &bytes[0..4] != VGLD_MAGIC {
            return Err(bad());
        }
        let mut h = [0u32; 12];
        for (i, slot) in h.iter_mut().enumerate() {
            let at = 4 + i * 4;
            *slot = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        }
        if h[0] != 1 {
            return Err(bad());
        }
        let (w, hh) = (h[1], h[2]);
        let (n_vit_h, n_vit_w, n_llm_h, n_llm_w) = (h[3], h[4], h[5], h[6]);
        let (pp, e, best_w, best_h) = (h[7], h[8], h[9], h[10]);
        // Size the three sections in u64 with checked maths so a corrupt
        // header cannot overflow into a short allocation.
        let rgb_len = u64::from(best_w)
            .checked_mul(u64::from(best_h))
            .and_then(|v| v.checked_mul(3));
        let pat_len = u64::from(n_vit_h)
            .checked_mul(u64::from(n_vit_w))
            .and_then(|v| v.checked_mul(u64::from(pp)))
            .and_then(|v| v.checked_mul(4));
        let emb_len = u64::from(n_llm_h)
            .checked_mul(u64::from(n_llm_w))
            .and_then(|v| v.checked_mul(u64::from(e)))
            .and_then(|v| v.checked_mul(4));
        let (Some(rgb_len), Some(pat_len), Some(emb_len)) = (rgb_len, pat_len, emb_len) else {
            return Err(bad());
        };
        let total = [rgb_len, pat_len, emb_len]
            .iter()
            .try_fold(VGLD_HEADER_BYTES as u64, |acc, v| acc.checked_add(*v));
        let short = || gate_error(format!("vision golden {}: short read", path.display()));
        if total.is_none_or(|t| (bytes.len() as u64) < t) {
            return Err(short());
        }
        let at = VGLD_HEADER_BYTES;
        let rgb = bytes[at..at + rgb_len as usize].to_vec();
        let at = at + rgb_len as usize;
        let patches = le_f32_slice(&bytes[at..at + pat_len as usize]);
        let at = at + pat_len as usize;
        let embeds = le_f32_slice(&bytes[at..at + emb_len as usize]);
        Ok(Self {
            w,
            h: hh,
            n_vit_h,
            n_vit_w,
            n_llm_h,
            n_llm_w,
            pp,
            e,
            best_w,
            best_h,
            rgb,
            patches,
            embeds,
        })
    }
}

fn le_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Per-row comparison of two [rows][e] matrices (gate :24-38): the minimum
/// and mean row cosine, the largest element difference, and the row the
/// minimum fell on.  f64 accumulation and the engine's exact formula
/// `ab / (sqrt(aa) * sqrt(bb) + 1e-30)`.
#[derive(Clone, Copy, Debug)]
pub struct RowStats {
    pub min_cos: f64,
    pub mean_cos: f64,
    pub max_abs: f64,
    pub worst: u32,
}

pub fn row_stats(a: &[f32], b: &[f32], rows: u32, e: u32) -> RowStats {
    let e = e as usize;
    let mut min_cos = 1.0f64;
    let mut max_abs = 0.0f64;
    let mut worst = 0u32;
    let mut sum = 0.0f64;
    for r in 0..rows as usize {
        let (mut aa, mut bb, mut ab) = (0.0f64, 0.0f64, 0.0f64);
        for j in 0..e {
            let x = f64::from(a[r * e + j]);
            let y = f64::from(b[r * e + j]);
            aa += x * x;
            bb += y * y;
            ab += x * y;
            let d = (x - y).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        let c = ab / (aa.sqrt() * bb.sqrt() + 1e-30);
        sum += c;
        if c < min_cos {
            min_cos = c;
            worst = r as u32;
        }
    }
    RowStats {
        min_cos,
        mean_cos: if rows > 0 { sum / f64::from(rows) } else { 0.0 },
        max_abs,
        worst,
    }
}

impl Model {
    /// The engine's `ds4_engine_v41_vision_golden` (gate :44-94): one image's
    /// preprocessing and the tower's aligner rows against the golden fixture.
    /// `Ok(true)` = every gating check passed (the engine's rc 0); `Ok(false)`
    /// = a threshold verdict failed and was printed; `Err` = an early refusal
    /// by name (no tower, bad fixture, failed plan or forward), the engine's
    /// early `return 1` paths.
    ///
    /// `meta` is the sidecar's own geometry (the engine reads the loaded
    /// config; the port reads the same bytes from the file the open bound).
    pub fn v41_vision_golden(
        &self,
        meta: &V41VisionMeta,
        image_path: &Path,
        golden_path: &Path,
    ) -> Result<bool> {
        // gate :45: refuse upfront when the open carries no tower.
        if unsafe { ds4_bridge_v41_has_vision(self.raw_ptr()) } == 0 {
            return Err(gate_error("this GGUF has no vision tower"));
        }
        let golden = V41Golden::read(golden_path)?;
        let image = std::fs::read(image_path).map_err(|e| {
            gate_error(format!("vision golden image {}: {e}", image_path.display()))
        })?;

        // (1) the size account (gate :59-65): the plan for the golden's own
        // image dimensions.
        let plan = v41_media::plan(golden.w, golden.h, &meta.config)
            .ok_or_else(|| gate_error("vision golden: plan failed"))?;
        let plan_ok = plan.n_vit_h == golden.n_vit_h
            && plan.n_vit_w == golden.n_vit_w
            && plan.n_llm_h == golden.n_llm_h
            && plan.n_llm_w == golden.n_llm_w
            && plan.best_w == golden.best_w
            && plan.best_h == golden.best_h;
        eprintln!(
            "[v41-golden] (1) plan {}: image {}x{} -> pad {}x{} (golden {}x{}), patch {}x{} (golden {}x{}), token {}x{} (golden {}x{})",
            if plan_ok { "ok" } else { "MISMATCH" },
            golden.w,
            golden.h,
            plan.best_w,
            plan.best_h,
            golden.best_w,
            golden.best_h,
            plan.n_vit_h,
            plan.n_vit_w,
            golden.n_vit_h,
            golden.n_vit_w,
            plan.n_llm_h,
            plan.n_llm_w,
            golden.n_llm_h,
            golden.n_llm_w,
        );
        if !plan_ok {
            return Err(gate_error(
                "vision golden: the plan does not match the golden grid",
            ));
        }
        // The patch width is not part of the plan.  A fixture whose pp is not
        // 3*patch*patch would make the native read past the golden slice (the
        // engine overreads instead of checking): refuse by name (port safety
        // guard, no engine counterpart).
        let pp = 3 * meta.config.patch * meta.config.patch;
        if golden.pp != pp {
            return Err(gate_error(format!(
                "vision golden: patch width {} is not 3*patch*patch ({pp})",
                golden.pp
            )));
        }

        // (2)/(3): the port's preprocessing, the production path
        // (v41_media::patches = decode + plan + PIL bicubic pad + patches).
        let p = v41_media::patches(&image, &meta.config)?;
        let mut ok = true;

        // (2) pixels, report only (see the module comment).
        {
            let mut nd = 0u64;
            let mut mx = 0u8;
            for (a, b) in p.padded_rgb.iter().zip(golden.rgb.iter()) {
                let d = a.abs_diff(*b);
                if d > 0 {
                    nd += 1;
                }
                if d > mx {
                    mx = d;
                }
            }
            eprintln!(
                "[v41-golden] (2) pixels (report only): {nd}/{} differ, max {mx} (PNG expected 0; JPEG decoder +-1..2, gate :34)",
                golden.rgb.len()
            );
        }

        // (3) patches vs the golden (gate :67-76).
        {
            let mut nd = 0u64;
            let mut mx = 0.0f64;
            for (a, b) in p.patches.iter().zip(golden.patches.iter()) {
                let d = (f64::from(*a) - f64::from(*b)).abs();
                if d > 0.0 {
                    nd += 1;
                }
                if d > mx {
                    mx = d;
                }
            }
            eprintln!(
                "[v41-golden] (3) patches: {nd}/{} elements differ, max {mx:.5} (about {:.1} pixel-levels)",
                golden.patches.len(),
                mx / (2.0 / 255.0)
            );
            if mx > VIS_GATE_PATCH_PIXELS {
                eprintln!(
                    "[v41-golden]     over 4 pixel-levels: not the decoder's +-1, the resize/pad transcription is wrong"
                );
                ok = false;
            }
        }

        // (4) the tower on the GOLDEN patches vs the golden embeds
        // (gate :79-85); this check is the tower alone.
        let n_llm = golden.n_llm();
        let e = golden.e;
        if e != self.shape().n_embd {
            return Err(gate_error(format!(
                "vision golden: embed width {e} is not the model's {}",
                self.shape().n_embd
            )));
        }
        let (mut olh, mut olw) = (0u32, 0u32);
        let rows = unsafe {
            ds4_bridge_v41_vision_run(
                self.raw_ptr(),
                golden.patches.as_ptr(),
                golden.patches.len() as u64,
                golden.n_vit_h,
                golden.n_vit_w,
                &mut olh,
                &mut olw,
            )
        };
        if rows.is_null() {
            return Err(gate_error("vision golden: tower forward failed"));
        }
        // The native allocates its own grid's rows; a golden grid that
        // disagrees would make the read below overrun -- refuse by name
        // (port safety guard, no engine counterpart).
        if u64::from(olh) * u64::from(olw) != n_llm {
            unsafe { ds4_bridge_v41_vision_rows_free(rows) };
            return Err(gate_error(format!(
                "vision golden: the tower produced {}x{} rows, the golden says {}x{}",
                olh, olw, golden.n_llm_h, golden.n_llm_w
            )));
        }
        let emb_g = unsafe { std::slice::from_raw_parts(rows, (n_llm * u64::from(e)) as usize) };
        let st = row_stats(emb_g, &golden.embeds, n_llm as u32, e);
        eprintln!(
            "[v41-golden] (4) tower (golden patches): {n_llm} rows x {e}, row cos mean {:.6} min {:.6} (row {}), max abs diff {:.5}",
            st.mean_cos, st.min_cos, st.worst, st.max_abs
        );
        if st.mean_cos < VIS_GATE_MEAN_COS || st.min_cos < VIS_GATE_MIN_COS {
            eprintln!(
                "[v41-golden]     a tower stage is wrong (mean < {VIS_GATE_MEAN_COS} or min row < {VIS_GATE_MIN_COS}; the official bf16 noise is in the threshold basis, gate :39-41)"
            );
            ok = false;
        }
        unsafe { ds4_bridge_v41_vision_rows_free(rows) };

        // (5) end-to-end on OUR patches, report only (gate :87-90); a failed
        // forward is skipped, exactly like the engine's `if (emb_o)`.
        let rows = unsafe {
            ds4_bridge_v41_vision_run(
                self.raw_ptr(),
                p.patches.as_ptr(),
                p.patches.len() as u64,
                p.plan.n_vit_h,
                p.plan.n_vit_w,
                &mut olh,
                &mut olw,
            )
        };
        if !rows.is_null() {
            let n = u64::from(olh) * u64::from(olw);
            if n == n_llm {
                let emb_o =
                    unsafe { std::slice::from_raw_parts(rows, (n * u64::from(e)) as usize) };
                let st = row_stats(emb_o, &golden.embeds, n_llm as u32, e);
                eprintln!(
                    "[v41-golden] (5) end-to-end (our patches): row cos mean {:.6} min {:.6} (row {}), max abs diff {:.5} (decoder differences ride here; report only)",
                    st.mean_cos, st.min_cos, st.worst, st.max_abs
                );
            }
            unsafe { ds4_bridge_v41_vision_rows_free(rows) };
        }

        // The engine's final verdict line (gate :91-92).
        eprintln!("vision-golden: {}", if ok { "ok" } else { "FAIL" });
        Ok(ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden_bytes(
        w: u32,
        h: u32,
        n_vit_h: u32,
        n_vit_w: u32,
        n_llm_h: u32,
        n_llm_w: u32,
        pp: u32,
        e: u32,
        best_w: u32,
        best_h: u32,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(VGLD_MAGIC);
        for v in [
            1u32, w, h, n_vit_h, n_vit_w, n_llm_h, n_llm_w, pp, e, best_w, best_h, 0,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend(std::iter::repeat_n(7u8, (best_w * best_h * 3) as usize));
        for i in 0..(n_vit_h * n_vit_w * pp) as usize {
            out.extend_from_slice(&(i as f32).to_le_bytes());
        }
        for i in 0..(n_llm_h * n_llm_w * e) as usize {
            out.extend_from_slice(&(i as f32).to_le_bytes());
        }
        out
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("v41-golden-{}-{name}.bin", std::process::id()))
    }

    #[test]
    fn golden_reader_reads_the_engine_format() {
        let path = temp_path("read");
        std::fs::write(&path, golden_bytes(1024, 768, 4, 3, 2, 1, 12, 5, 56, 42)).unwrap();
        let g = V41Golden::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!((g.w, g.h), (1024, 768));
        assert_eq!((g.n_vit_h, g.n_vit_w), (4, 3));
        assert_eq!((g.n_llm_h, g.n_llm_w), (2, 1));
        assert_eq!((g.pp, g.e), (12, 5));
        assert_eq!((g.best_w, g.best_h), (56, 42));
        assert_eq!(g.rgb.len(), 56 * 42 * 3);
        assert!(g.rgb.iter().all(|&v| v == 7));
        assert_eq!(g.patches.len(), 12 * 12);
        assert_eq!(g.patches[1], 1.0);
        assert_eq!(g.embeds.len(), 2 * 5);
        assert_eq!(g.embeds[9], 9.0);
        assert_eq!(g.n_vit(), 12);
        assert_eq!(g.n_llm(), 2);
    }

    #[test]
    fn golden_reader_refuses_bad_magic_version_and_short() {
        let path = temp_path("bad");
        std::fs::write(&path, b"XGLD").unwrap();
        assert!(V41Golden::read(&path).is_err());

        let mut bytes = golden_bytes(8, 8, 1, 1, 1, 1, 4, 2, 14, 14);
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes()); // ver=2
        std::fs::write(&path, &bytes).unwrap();
        assert!(V41Golden::read(&path).is_err());

        let mut bytes = golden_bytes(8, 8, 1, 1, 1, 1, 4, 2, 14, 14);
        bytes.truncate(bytes.len() - 4);
        std::fs::write(&path, &bytes).unwrap();
        assert!(V41Golden::read(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn row_stats_reproduces_the_engine_formula() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let same = row_stats(&a, &a, 2, 2);
        assert!((same.mean_cos - 1.0).abs() < 1e-12);
        assert!((same.min_cos - 1.0).abs() < 1e-12);
        assert_eq!(same.max_abs, 0.0);
        assert_eq!(same.worst, 0);

        // Row 1 orthogonal: cos 0, the minimum, max abs diff 4.
        let b = [1.0f32, 2.0, -4.0, 3.0];
        let st = row_stats(&a, &b, 2, 2);
        assert_eq!(st.worst, 1);
        assert!(st.min_cos.abs() < 1e-12);
        assert!((st.mean_cos - 0.5).abs() < 1e-12);
        assert!((st.max_abs - 7.0).abs() < 1e-12);
    }

    #[test]
    fn row_stats_single_row_negation_is_minus_one() {
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        let st = row_stats(&a, &b, 1, 2);
        assert!((st.min_cos + 1.0).abs() < 1e-12);
        assert_eq!(st.worst, 0);
    }
}
