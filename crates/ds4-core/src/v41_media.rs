//! DeepSeek V4.1 image preprocessing: bytes -> ViT patches.
//!
//! 1:1 port of the engine's `src/core/core_v41_image.c` (the official
//! `inference/image_processor.py`, transcribed expression by expression):
//!   1. the size plan (`v41_image_plan`): upscale a small image to `min_pixels`,
//!      round up to a patch multiple, and if the span still exceeds
//!      `max_image_tokens` solve the largest aspect-preserving size that fits;
//!   2. `ImageOps.pad` resize (`pil_*`): PIL's bicubic is a separable
//!      anti-aliased convolution with 22-bit fixed-point accumulation and
//!      8-bit truncation; same image => same pixels as PIL (decode is the
//!      `image` crate, so JPEG is within the decoder's own +-1, not this code);
//!   3. normalization `(x/255 - 0.5)/0.5` rounded to bf16, laid out in the
//!      official reshape/permute order `[patch][c][ph][pw]`.
//!
//! Every constant and step cites the engine line it mirrors. The values are
//! geometry and rounding only; no model maths lives here (the tower is native).

use crate::{Error, Result};
use image::RgbImage;

/// The V4.1 vision-tower geometry, read from the sidecar's `deepseek4.vision.*`
/// metadata (`core_validate_v41.c:48-66`). Carried as data because the tower
/// config is metadata truth, not a compile-time constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V41VisionConfig {
    pub patch: u32,      // deepseek4.vision.patch_size (14)
    pub downsample: u32, // deepseek4.vision.downsample_ratio (3)
    pub max_tokens: u32, // deepseek4.vision.max_image_tokens (1024)
    pub min_pixels: u32, // deepseek4.vision.min_pixels (295936)
}

/// The padded grid and the LLM token grid for one image (`core_v41_image.c:25`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V41ImagePlan {
    pub n_vit_h: u32, // patch rows
    pub n_vit_w: u32, // patch columns
    pub n_llm_h: u32, // aligner rows (n_vit_h / downsample, rounded up)
    pub n_llm_w: u32,
    pub best_w: u32, // padded width, a multiple of patch_size
    pub best_h: u32,
}

impl V41ImagePlan {
    /// Prompt span length: one image_start, then per LLM row `n_llm_w` aligner
    /// rows plus an image_newline, then image_end (`core_v41_image.c:23`).
    pub const fn span_tokens(&self) -> u32 {
        self.n_llm_h * (self.n_llm_w + 1) + 2
    }
}

/// The preprocessing output for one image: the plan, the padded RGB (kept for
/// the gate's pixel check), and the patches in `[patch][c][ph][pw]` order.
pub struct V41Patches {
    pub plan: V41ImagePlan,
    pub padded_rgb: Vec<u8>, // [best_h][best_w][3]
    pub patches: Vec<f32>,   // [n_vit_h*n_vit_w][3*patch*patch]
}

fn ceil_div(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}

/// Size plan (`core_v41_image.c:25-52`). `None` when the image cannot fit.
pub fn plan(w0: u32, h0: u32, cfg: &V41VisionConfig) -> Option<V41ImagePlan> {
    let p = u64::from(cfg.patch);
    let r = u64::from(cfg.downsample);
    let maxt = u64::from(cfg.max_tokens);
    if w0 == 0 || h0 == 0 || p == 0 || r == 0 || maxt < 4 {
        return None;
    }
    // Python ints: W/H start integer and the upscale truncates toward zero.
    let mut w = i64::from(w0);
    let mut h = i64::from(h0);
    if (w as f64) * (h as f64) < f64::from(cfg.min_pixels) {
        let ratio = (f64::from(cfg.min_pixels) / ((w as f64) * (h as f64))).sqrt();
        w = ((w as f64) * ratio) as i64;
        h = ((h as f64) * ratio) as i64;
    }
    let mut bw = ceil_div(w as u64, p) * p;
    let mut bh = ceil_div(h as u64, p) * p;
    let mut lh = ceil_div(bh / p, r);
    let mut lw = ceil_div(bw / p, r);
    if lh * (lw + 1) + 2 > maxt {
        // solve_resize_ratio: the largest aspect-preserving size that fits.
        let rr = h as f64 / w as f64;
        let max_w = ((maxt - 2) as f64 / rr + 0.25).sqrt() - 0.5;
        let max_h = max_w * rr;
        let cell = p * r;
        if max_w < 1.0 {
            bh = (maxt - 2) / 2 * cell;
            bw = cell;
        } else if max_h < 1.0 {
            bh = cell;
            bw = (maxt - 3) * cell;
        } else {
            let bw_ = max_w.floor() * cell as f64 / w as f64;
            let bh_ = max_h.floor() * cell as f64 / h as f64;
            let beta = if bw_ < bh_ { bw_ } else { bh_ };
            bh = (h as f64 * beta / p as f64).floor() as u64 * p;
            bw = (w as f64 * beta / p as f64).floor() as u64 * p;
        }
        lh = ceil_div(bh / p, r);
        lw = ceil_div(bw / p, r);
        if lh * (lw + 1) + 2 > maxt {
            return None;
        }
    }
    if bw == 0 || bh == 0 {
        return None;
    }
    Some(V41ImagePlan {
        n_vit_h: (bh / p) as u32,
        n_vit_w: (bw / p) as u32,
        n_llm_h: lh as u32,
        n_llm_w: lw as u32,
        best_w: bw as u32,
        best_h: bh as u32,
    })
}

// ---- PIL bicubic resample, transcribed (libImaging/Resample.c 8bpc path) ----

const PIL_PRECISION_BITS: u32 = 22;

fn pil_bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct PilCoeffs {
    bounds: Vec<(i32, i32)>, // (start, count) per output coordinate
    k: Vec<i32>,             // quantized to PIL_PRECISION_BITS, out_size*ksize
    ksize: usize,
}

/// Per-output [start, count] and 22-bit fixed-point coefficients, with the
/// anti-aliasing kernel widening on downscale (`core_v41_image.c:67-92`).
fn pil_coeffs(in_size: u32, out_size: u32) -> PilCoeffs {
    let scale = f64::from(in_size) / f64::from(out_size);
    let fscale = if scale < 1.0 { 1.0 } else { scale };
    let support = 2.0 * fscale;
    let ksize = (support.ceil() as usize) * 2 + 1;
    let mut kd = vec![0.0f64; out_size as usize * ksize];
    let mut bounds = vec![(0i32, 0i32); out_size as usize];
    for xx in 0..out_size as usize {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / fscale;
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size as i32 {
            xmax = in_size as i32;
        }
        let count = xmax - xmin;
        let base = xx * ksize;
        let mut ww = 0.0;
        for x in 0..count as usize {
            let v = pil_bicubic((x as f64 + xmin as f64 + 0.5 - center) * ss);
            kd[base + x] = v;
            ww += v;
        }
        if ww != 0.0 {
            for x in 0..count as usize {
                kd[base + x] /= ww;
            }
        }
        bounds[xx] = (xmin, count);
    }
    // normalize_coeffs_8bpc: round-to-nearest in the fixed-point domain.
    let k = kd
        .iter()
        .map(|&v| {
            let q = v * (1i64 << PIL_PRECISION_BITS) as f64;
            if q < 0.0 {
                (q - 0.5) as i32
            } else {
                (q + 0.5) as i32
            }
        })
        .collect();
    PilCoeffs { bounds, k, ksize }
}

/// Truncating arithmetic shift + clip, same as PIL's clip8
/// (`core_v41_image.c:94-97`). Accumulated in i64 so no debug overflow is
/// possible; on 8bpc inputs the value is identical to the engine's int32 sum.
fn pil_clip8(ss: i64) -> u8 {
    let v = ss >> PIL_PRECISION_BITS;
    v.clamp(0, 255) as u8
}

/// Horizontal then vertical pass (`ImagingResampleInner` order); a dimension
/// that does not change is skipped, as PIL does (`core_v41_image.c:99-127`).
fn pil_resize_rgb(input: &[u8], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    let mut cur = input.to_vec();
    let mut cw = w;
    let half: i64 = 1 << (PIL_PRECISION_BITS - 1);
    if tw != w {
        let c = pil_coeffs(w, tw);
        let mut out = vec![0u8; tw as usize * h as usize * 3];
        for y in 0..h as usize {
            for xx in 0..tw as usize {
                let (xmin, count) = c.bounds[xx];
                let k = &c.k[xx * c.ksize..xx * c.ksize + c.ksize];
                let (mut s0, mut s1, mut s2) = (half, half, half);
                for x in 0..count as usize {
                    let px = (y * w as usize + (x + xmin as usize)) * 3;
                    s0 += i64::from(cur[px]) * i64::from(k[x]);
                    s1 += i64::from(cur[px + 1]) * i64::from(k[x]);
                    s2 += i64::from(cur[px + 2]) * i64::from(k[x]);
                }
                let o = (y * tw as usize + xx) * 3;
                out[o] = pil_clip8(s0);
                out[o + 1] = pil_clip8(s1);
                out[o + 2] = pil_clip8(s2);
            }
        }
        cur = out;
        cw = tw;
    }
    if th != h {
        let c = pil_coeffs(h, th);
        let mut out = vec![0u8; cw as usize * th as usize * 3];
        for yy in 0..th as usize {
            let (ymin, count) = c.bounds[yy];
            let k = &c.k[yy * c.ksize..yy * c.ksize + c.ksize];
            for x in 0..cw as usize {
                let (mut s0, mut s1, mut s2) = (half, half, half);
                for y in 0..count as usize {
                    let px = ((y + ymin as usize) * cw as usize + x) * 3;
                    s0 += i64::from(cur[px]) * i64::from(k[y]);
                    s1 += i64::from(cur[px + 1]) * i64::from(k[y]);
                    s2 += i64::from(cur[px + 2]) * i64::from(k[y]);
                }
                let o = (yy * cw as usize + x) * 3;
                out[o] = pil_clip8(s0);
                out[o + 1] = pil_clip8(s1);
                out[o + 2] = pil_clip8(s2);
            }
        }
        cur = out;
    }
    cur
}

/// Python round(): ties to even (`core_v41_image.c:128`).
fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// `ImageOps.contain` + `ImageOps.pad`: scale to fit, then center on a
/// (127,127,127) background (`core_v41_image.c:130-148`).
fn resize_pad(rgb: &[u8], w: u32, h: u32, best_w: u32, best_h: u32) -> Vec<u8> {
    let mut tw = best_w;
    let mut th = best_h;
    let im_ratio = f64::from(w) / f64::from(h);
    let dest_ratio = f64::from(best_w) / f64::from(best_h);
    if im_ratio != dest_ratio {
        if im_ratio > dest_ratio {
            let nh = py_round(f64::from(h) / f64::from(w) * f64::from(best_w));
            if nh != i64::from(best_h) {
                th = if nh > 0 { nh as u32 } else { 1 };
            }
        } else {
            let nw = py_round(f64::from(w) / f64::from(h) * f64::from(best_h));
            if nw != i64::from(best_w) {
                tw = if nw > 0 { nw as u32 } else { 1 };
            }
        }
    }
    let rs = pil_resize_rgb(rgb, w, h, tw, th);
    if tw == best_w && th == best_h {
        return rs;
    }
    let mut out = vec![127u8; best_w as usize * best_h as usize * 3];
    let x0 = if tw != best_w {
        py_round(f64::from(best_w - tw) * 0.5) as u32
    } else {
        0
    };
    let y0 = if tw != best_w {
        0
    } else {
        py_round(f64::from(best_h - th) * 0.5) as u32
    };
    for y in 0..th as usize {
        let dst = ((y + y0 as usize) * best_w as usize + x0 as usize) * 3;
        let src = y * tw as usize * 3;
        out[dst..dst + tw as usize * 3].copy_from_slice(&rs[src..src + tw as usize * 3]);
    }
    out
}

/// f32 -> nearest bf16 grid point, returned as f32 (the engine's
/// `ds4_bf16_round_f32`, src/common/ds4_float.h:66-73: round half to even).
fn bf16_round(f: f32) -> f32 {
    let u = f.to_bits();
    if u & 0x7f80_0000 == 0x7f80_0000 {
        // NaN / Inf: keep the top 16 bits, set the quiet bit for a NaN payload.
        let s = (u >> 16) | if u & 0xffff != 0 { 0x40 } else { 0 };
        return f32::from_bits(s << 16);
    }
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

fn decode_rgb(bytes: &[u8]) -> Result<RgbImage> {
    let image = image::load_from_memory(bytes)
        .map_err(|e| Error {
            code: 1,
            message: format!("V4.1 image decode failed: {e}"),
        })?
        .into_rgb8();
    if image.width() == 0 || image.height() == 0 {
        return Err(Error {
            code: 1,
            message: "V4.1 image has zero extent".into(),
        });
    }
    Ok(image)
}

/// Decode, plan, resize-pad, and cut patches (`core_v41_image.c:150-172`).
pub fn patches(bytes: &[u8], cfg: &V41VisionConfig) -> Result<V41Patches> {
    let img = decode_rgb(bytes)?;
    let (w, h) = (img.width(), img.height());
    let plan = plan(w, h, cfg).ok_or_else(|| Error {
        code: 1,
        message: format!("image {w}x{h} does not fit the {}-token size budget", cfg.max_tokens),
    })?;
    let px = resize_pad(img.as_raw(), w, h, plan.best_w, plan.best_h);
    let p = cfg.patch as usize;
    let (nh, nw, bw) = (plan.n_vit_h as usize, plan.n_vit_w as usize, plan.best_w as usize);
    let pp = p * p;
    let mut out = vec![0f32; nh * nw * 3 * pp];
    for i in 0..nh {
        for j in 0..nw {
            let base = (i * nw + j) * 3 * pp;
            for c in 0..3 {
                for ph in 0..p {
                    for pw in 0..p {
                        let x = f32::from(px[((i * p + ph) * bw + (j * p + pw)) * 3 + c]) / 255.0;
                        out[base + c * pp + ph * p + pw] = bf16_round((x - 0.5) / 0.5);
                    }
                }
            }
        }
    }
    Ok(V41Patches {
        plan,
        padded_rgb: px,
        patches: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> V41VisionConfig {
        // The shipped sidecar's own values (see the handoff artifact contract).
        V41VisionConfig {
            patch: 14,
            downsample: 3,
            max_tokens: 1024,
            min_pixels: 295936,
        }
    }

    #[test]
    fn plan_snaps_to_patch_and_downsample_grids() {
        let c = cfg();
        let p = plan(1024, 768, &c).expect("fits");
        assert_eq!(p.best_w % c.patch, 0);
        assert_eq!(p.best_h % c.patch, 0);
        assert_eq!(p.n_vit_w, p.best_w / c.patch);
        assert_eq!(p.n_llm_w, p.n_vit_w.div_ceil(c.downsample));
        assert!(p.span_tokens() <= c.max_tokens);
    }

    #[test]
    fn plan_upscales_tiny_images_to_min_pixels() {
        let c = cfg();
        let p = plan(64, 64, &c).expect("fits");
        // 64*64 is far below min_pixels, so the plan is scaled up.
        assert!(u64::from(p.best_w) * u64::from(p.best_h) > 64 * 64);
        assert!(p.span_tokens() <= c.max_tokens);
    }

    #[test]
    fn plan_huge_image_shrinks_under_max_tokens() {
        let c = cfg();
        let p = plan(8000, 6000, &c).expect("fits after solve_resize_ratio");
        assert!(p.span_tokens() <= c.max_tokens);
        assert_eq!(p.best_w % c.patch, 0);
        assert_eq!(p.best_h % c.patch, 0);
    }

    #[test]
    fn span_tokens_formula_matches_engine() {
        let p = V41ImagePlan {
            n_vit_h: 48,
            n_vit_w: 48,
            n_llm_h: 16,
            n_llm_w: 16,
            best_w: 672,
            best_h: 672,
        };
        assert_eq!(p.span_tokens(), 16 * 17 + 2);
    }

    #[test]
    fn bf16_round_matches_half_to_even() {
        // 1.0 is exact on the bf16 grid; 1.0 + 2^-9 rounds to the even neighbor.
        assert_eq!(bf16_round(1.0), 1.0);
        let tiny = 1.0 + 2f32.powi(-9);
        assert!(bf16_round(tiny) == 1.0 || bf16_round(tiny) == 1.0 + 2f32.powi(-8));
    }

    #[test]
    fn patches_have_the_expected_shape_and_range() {
        let c = cfg();
        // A small solid PNG; the plan upscales it, and every patch value is a
        // normalized bf16 value within [-1, 1].
        let img = RgbImage::from_pixel(100, 80, image::Rgb([200, 100, 50]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encode png");
        let out = patches(&png, &c).expect("patch");
        let pp = (c.patch * c.patch) as usize;
        assert_eq!(out.patches.len(), out.plan.n_vit_h as usize * out.plan.n_vit_w as usize * 3 * pp);
        assert!(out.patches.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }
}
