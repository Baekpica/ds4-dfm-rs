//! Vision-tower image input for Qwen4Exp, Ling-3.0-VL and GLM-5.3: decode,
//! geometry, Pillow-style bicubic resize, normalization and patch order.
//! Native receives the packed patch rows and adds only the tower's position
//! inputs. The arithmetic is the C it replaces (ds4.c image probes,
//! `qwen4exp_resize_rgb_pillow` and the three packers), including C's
//! resize-weight rounding, `round(q ± 0.5)`, where Pillow truncates q + 0.5.
//! PNG decodes to stb_image's bytes. JPEG goes through libjpeg-turbo with
//! libjpeg's defaults (ISLOW IDCT, fancy upsampling): PIL's decoder and
//! settings, so pixels match the HF processors where stb_image's differ by
//! a level or more.
//!
//! ```text
//! bytes -> RGB8 (+ GLM EXIF orientation) -> resize -> patches [rows, dim]
//! rows in 2x2 merge order:   block (y, x) -> (0,0) (0,1) (1,0) (1,1)
//! ```

use image::{DynamicImage, ImageReader};
use std::io::Cursor;

const SOURCE_MAX_PIXELS: u64 = 16_777_216;
const ASPECT_MAX: u64 = 200;
const QWEN_FACTOR: u32 = 32;
const QWEN_MIN_PIXELS: u64 = 65_536;
const LING_MIN_PIXELS: u64 = 4_096;
const LING_MAX_PIXELS: u64 = 4_194_304;
const VIT_PATCH: u32 = 16;
const GLM_PATCH: u32 = 14;
const GLM_FACTOR: u32 = 28;
const GLM_MAX_SIDE: u32 = 16_384;
const GLM_MAX_BYTES: usize = 64 << 20;
const GLM_MAX_TOKENS: u64 = 8000;
/// OpenAI CLIP normalization (GLM and Ling); Qwen scales to [-1, 1]. The
/// processors' literals, which round to the same f32 as the C ones.
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

const RESIZE_PRECISION: u32 = 22;
/// SOI followed by a marker.
const JPEG_MAGIC: [u8; 3] = [0xFF, 0xD8, 0xFF];
/// The C hash's basis, one digit short of FNV-64's 14695981039346656037;
/// kept because the hash keys persisted disk KV.
const FNV_OFFSET: u64 = 1_469_598_103_934_665_603;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Geometry of one image after the family's smart resize.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImagePlan {
    pub source_width: u32,
    pub source_height: u32,
    /// Resized pixels; GLM pads them to `padded_*`.
    pub content_width: u32,
    pub content_height: u32,
    pub padded_width: u32,
    pub padded_height: u32,
    pub grid_h: u32,
    pub grid_w: u32,
    /// Merged rows the tower emits: grid / 4.
    pub token_count: u32,
}

/// Which vision tower an image is packed for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tower {
    Qwen,
    Ling,
    Glm,
}

pub struct Rgb {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Header-only geometry, as the C probe read it with `stbi_info`.
pub fn probe(tower: Tower, data: &[u8]) -> Result<ImagePlan, String> {
    let (width, height) = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .into_dimensions()
        .map_err(|e| format!("cannot read PNG/JPEG dimensions: {e}"))?;
    match tower {
        Tower::Qwen => plan_qwen(width, height, QWEN_MIN_PIXELS, SOURCE_MAX_PIXELS),
        Tower::Ling => plan_qwen(width, height, LING_MIN_PIXELS, LING_MAX_PIXELS),
        Tower::Glm => plan_glm(data, width, height),
    }
}

/// The Qwen3-VL smart resize (factor 32, patch 16, 2x2 merge) under each
/// processor's pixel budget.
fn plan_qwen(width: u32, height: u32, min: u64, max: u64) -> Result<ImagePlan, String> {
    let source = u64::from(width) * u64::from(height);
    if width == 0 || height == 0 {
        return Err("cannot read PNG/JPEG dimensions: empty image".into());
    }
    if source > SOURCE_MAX_PIXELS {
        return Err("source image exceeds 16,777,216 pixels".into());
    }
    if u64::from(width.max(height)) > u64::from(width.min(height)) * ASPECT_MAX {
        return Err("image aspect ratio must not exceed 200:1".into());
    }

    let factor = f64::from(QWEN_FACTOR);
    let snap = |v: f64| (v as u32).saturating_mul(QWEN_FACTOR).max(QWEN_FACTOR);
    let (h, w) = (f64::from(height), f64::from(width));
    // lrint: round half to even, as C's default rounding mode.
    let mut rh = snap((h / factor).round_ties_even());
    let mut rw = snap((w / factor).round_ties_even());
    let resized = u64::from(rh) * u64::from(rw);
    if resized > max {
        let beta = (source as f64 / max as f64).sqrt();
        rh = snap((h / beta / factor).floor());
        rw = snap((w / beta / factor).floor());
    } else if resized < min {
        let beta = (min as f64 / source as f64).sqrt();
        rh = ((h * beta / factor).ceil() as u32) * QWEN_FACTOR;
        rw = ((w * beta / factor).ceil() as u32) * QWEN_FACTOR;
    }

    let (grid_h, grid_w) = (rh / VIT_PATCH, rw / VIT_PATCH);
    let tokens = u64::from(grid_h) * u64::from(grid_w) / 4;
    if grid_h % 2 != 0 || grid_w % 2 != 0 || tokens == 0 || tokens > u64::from(u32::MAX) {
        return Err("invalid image grid".into());
    }
    Ok(ImagePlan {
        source_width: width,
        source_height: height,
        content_width: rw,
        content_height: rh,
        padded_width: rw,
        padded_height: rh,
        grid_h,
        grid_w,
        token_count: tokens as u32,
    })
}

/// GLM-5.3: EXIF-oriented size, factor-28 smart resize, content scaled into
/// the padded canvas (ds4.c glm53_vision_probe).
fn plan_glm(data: &[u8], width: u32, height: u32) -> Result<ImagePlan, String> {
    if data.len() > GLM_MAX_BYTES
        || width == 0
        || height == 0
        || width > GLM_MAX_SIDE
        || height > GLM_MAX_SIDE
        || u64::from(width) * u64::from(height) > SOURCE_MAX_PIXELS
    {
        return Err("invalid or oversized PNG/JPEG image".into());
    }
    let (width, height) = match jpeg_orientation(data).transposes() {
        true => (height, width),
        false => (width, height),
    };
    let (padded_h, padded_w) = glm_smart_resize(height, width);
    let mut scale =
        (f64::from(padded_h) / f64::from(height)).min(f64::from(padded_w) / f64::from(width));
    let min_pixels = 16u64 * 2 * u64::from(GLM_FACTOR * GLM_FACTOR);
    if 2 * u64::from(height) * u64::from(width) >= min_pixels && scale > 1.0 {
        scale = 1.0;
    }
    let content_h = ((f64::from(height) * scale).floor() as u32).clamp(1, padded_h);
    let content_w = ((f64::from(width) * scale).floor() as u32).clamp(1, padded_w);
    let (grid_h, grid_w) = (padded_h / GLM_PATCH, padded_w / GLM_PATCH);
    let tokens = u64::from(grid_h) * u64::from(grid_w) / 4;
    if grid_h % 2 != 0 || grid_w % 2 != 0 || tokens == 0 || tokens > GLM_MAX_TOKENS {
        return Err("invalid GLM image grid".into());
    }
    Ok(ImagePlan {
        source_width: width,
        source_height: height,
        content_width: content_w,
        content_height: content_h,
        padded_width: padded_w,
        padded_height: padded_h,
        grid_h,
        grid_w,
        token_count: tokens as u32,
    })
}

fn glm_smart_resize(height: u32, width: u32) -> (u32, u32) {
    let factor = GLM_FACTOR;
    let up = |v: u32| v.div_ceil(factor) * factor;
    let min_pixels = 16u64 * 2 * u64::from(factor * factor);
    let max_pixels = 8000u64 * 2 * u64::from(factor * factor);
    let (mut h, mut w) = (up(height), up(width));
    let mut budget = 2 * u64::from(h) * u64::from(w);
    if budget < min_pixels {
        let scale = (min_pixels as f64 / (2.0 * f64::from(height) * f64::from(width))).sqrt();
        h = up((f64::from(height) * scale).ceil() as u32);
        w = up((f64::from(width) * scale).ceil() as u32);
        budget = 2 * u64::from(h) * u64::from(w);
    }
    if budget > max_pixels {
        // Largest height whose aspect-kept width still fits the budget.
        let (mut low, mut high) = (1u32, height);
        h = factor;
        w = factor;
        while low <= high {
            let ch = low + (high - low) / 2;
            let cw = ((f64::from(width) * f64::from(ch) / f64::from(height)).floor() as u32).max(1);
            let (candidate_h, candidate_w) = (up(ch), up(cw));
            if 2 * u64::from(candidate_h) * u64::from(candidate_w) <= max_pixels {
                h = candidate_h;
                w = candidate_w;
                low = ch + 1;
            } else {
                high = ch - 1;
            }
        }
    }
    (h, w)
}

/// PNG/JPEG to RGB8. JPEG as PIL decodes it; PNG as stb_image's 3-channel
/// load: alpha dropped, gray replicated, 16-bit samples truncated to their
/// high byte.
pub fn decode(data: &[u8]) -> Result<Rgb, String> {
    if let Some(rgb) = decode_jpeg(data)? {
        return Ok(rgb);
    }
    let image = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| format!("PNG/JPEG pixel decode failed: {e}"))?;
    let (width, height) = (image.width(), image.height());
    let data = match image {
        DynamicImage::ImageRgb16(_)
        | DynamicImage::ImageRgba16(_)
        | DynamicImage::ImageLuma16(_)
        | DynamicImage::ImageLumaA16(_) => image
            .into_rgb16()
            .into_raw()
            .into_iter()
            .map(|v| (v >> 8) as u8)
            .collect(),
        _ => image.into_rgb8().into_raw(),
    };
    Ok(Rgb {
        width,
        height,
        data,
    })
}

/// A JPEG's RGB through libjpeg-turbo; None for other formats.
fn decode_jpeg(data: &[u8]) -> Result<Option<Rgb>, String> {
    if !data.starts_with(&JPEG_MAGIC) {
        return Ok(None);
    }
    let fail = |e: turbojpeg::Error| format!("PNG/JPEG pixel decode failed: {e}");
    let header = turbojpeg::read_header(data).map_err(fail)?;
    // ponytail: libjpeg-turbo cannot turn CMYK/YCCK into RGB and PIL's own
    // CMYK conversion is not ported; the image crate decodes those.
    if matches!(
        header.colorspace,
        turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
    ) {
        return Ok(None);
    }
    let image = turbojpeg::decompress(data, turbojpeg::PixelFormat::RGB).map_err(fail)?;
    Ok(Some(Rgb {
        width: image.width as u32,
        height: image.height as u32,
        data: image.pixels,
    }))
}

/// Qwen's KV-cache identity of an image: FNV-1a over the little-endian
/// size and the decoded RGB bytes.
pub fn pixel_hash(rgb: &Rgb) -> u64 {
    let mut h = FNV_OFFSET;
    let size = rgb
        .width
        .to_le_bytes()
        .into_iter()
        .chain(rgb.height.to_le_bytes());
    for byte in size.chain(rgb.data.iter().copied()) {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Decode, orient (GLM), resize and pack one image for `tower`.
pub fn pack(tower: Tower, data: &[u8], plan: &ImagePlan) -> Result<Vec<f32>, String> {
    let mut rgb = decode(data)?;
    if tower == Tower::Glm {
        rgb = orient(rgb, jpeg_orientation(data));
    }
    if rgb.width != plan.source_width || rgb.height != plan.source_height {
        return Err("PNG/JPEG decode changed the image size".into());
    }
    let resized = resize(&rgb, plan.content_width, plan.content_height);
    Ok(match tower {
        Tower::Qwen => pack_rows(
            plan,
            VIT_PATCH,
            2,
            |v, _| f32::from(v).mul_add(2.0 / 255.0, -1.0),
            &resized,
        ),
        Tower::Ling => pack_rows(plan, VIT_PATCH, 1, clip, &resized),
        Tower::Glm => pack_rows(plan, GLM_PATCH, 2, clip, &resized),
    })
}

/// The C build (-O3 -ffast-math) computes v * (1/255) - mean fused, then
/// multiplies by 1/std; Qwen's v * (2/255) - 1 is fused as well. Same here,
/// so Qwen and Ling rows match it bit for bit; GLM's vectorized C loop fuses
/// only some lanes, so its rows differ by an ulp at some positions.
fn clip(v: u8, c: usize) -> f32 {
    f32::from(v).mul_add(1.0 / 255.0, -CLIP_MEAN[c]) * (1.0 / CLIP_STD[c])
}

/// Rows in 2x2 merge order; per row: channel, frame, y, x. Pixels outside
/// the content (GLM's padding) read as 0 before normalization. Each patch
/// line is written contiguously and later frames copy the first.
fn pack_rows(
    plan: &ImagePlan,
    patch: u32,
    frames: usize,
    norm: impl Fn(u8, usize) -> f32,
    rgb: &Rgb,
) -> Vec<f32> {
    let p = patch as usize;
    let plane = p * p;
    let dim = 3 * frames * plane;
    let rows = (plan.grid_h * plan.grid_w) as usize;
    let (w, h) = (rgb.width as usize, rgb.height as usize);
    let lut: [[f32; 256]; 3] = std::array::from_fn(|c| std::array::from_fn(|v| norm(v as u8, c)));
    let mut out = vec![0f32; rows * dim];
    let mut dst_rows = out.chunks_exact_mut(dim);
    for by in 0..(plan.grid_h / 2) as usize {
        for bx in 0..(plan.grid_w / 2) as usize {
            for (my, mx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let dst = dst_rows.next().expect("grid rows");
                let (py, px) = ((2 * by + my) * p, (2 * bx + mx) * p);
                let inside = p.min(w.saturating_sub(px));
                for (c, lut) in lut.iter().enumerate() {
                    let base = c * frames * plane;
                    for y in 0..p {
                        let line = &mut dst[base + y * p..base + (y + 1) * p];
                        let src = (py + y < h && inside > 0)
                            .then(|| &rgb.data[((py + y) * w + px) * 3..]);
                        let n = if src.is_some() { inside } else { 0 };
                        if let Some(src) = src {
                            for (d, px) in line[..n].iter_mut().zip(src.as_chunks::<3>().0) {
                                *d = lut[px[c] as usize];
                            }
                        }
                        line[n..].fill(lut[0]);
                    }
                    for f in 1..frames {
                        dst.copy_within(base..base + plane, base + f * plane);
                    }
                }
            }
        }
    }
    out
}

/// Separable antialiased bicubic (a = -0.5) in 22-bit fixed point; the
/// horizontal pass runs first, as in Pillow. The vertical pass accumulates
/// whole rows (exact integer sums, so the order is free).
pub fn resize(src: &Rgb, width: u32, height: u32) -> Rgb {
    if src.width == width && src.height == height {
        return Rgb {
            width,
            height,
            data: src.data.clone(),
        };
    }
    let horizontal = Axis::new(src.width, width);
    let vertical = Axis::new(src.height, height);
    let (sw, w) = (src.width as usize, width as usize);
    let round = 1i64 << (RESIZE_PRECISION - 1);
    let mut tmp = vec![0u8; w * src.height as usize * 3];
    for (row, out) in src
        .data
        .chunks_exact(sw * 3)
        .zip(tmp.chunks_exact_mut(w * 3))
    {
        for (x, out) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (first, taps) = horizontal.taps(x);
            let mut acc = [round; 3];
            for (&weight, px) in taps.iter().zip(row[first * 3..].as_chunks::<3>().0) {
                let weight = i64::from(weight);
                for (a, &v) in acc.iter_mut().zip(px) {
                    *a += i64::from(v) * weight;
                }
            }
            for (o, a) in out.iter_mut().zip(acc) {
                *o = clip_fixed(a);
            }
        }
    }
    let mut data = vec![0u8; w * height as usize * 3];
    let mut acc = vec![0i64; w * 3];
    for (y, out) in data.chunks_exact_mut(w * 3).enumerate() {
        let (first, taps) = vertical.taps(y);
        acc.fill(round);
        for (k, &weight) in taps.iter().enumerate() {
            let row = &tmp[(first + k) * w * 3..(first + k + 1) * w * 3];
            let weight = i64::from(weight);
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += i64::from(v) * weight;
            }
        }
        for (o, &a) in out.iter_mut().zip(&acc) {
            *o = clip_fixed(a);
        }
    }
    Rgb {
        width,
        height,
        data,
    }
}

fn clip_fixed(acc: i64) -> u8 {
    (acc >> RESIZE_PRECISION).clamp(0, 255) as u8
}

/// One axis's taps: first input index and weights per output index.
struct Axis {
    ksize: usize,
    first: Vec<usize>,
    count: Vec<usize>,
    weights: Vec<i32>,
}

impl Axis {
    fn new(input: u32, output: u32) -> Self {
        let scale = f64::from(input) / f64::from(output);
        let filter_scale = scale.max(1.0);
        let support = 2.0 * filter_scale;
        let ksize = support.ceil() as usize * 2 + 1;
        let fixed_scale = f64::from(1u32 << RESIZE_PRECISION);
        let mut axis = Self {
            ksize,
            first: Vec::with_capacity(output as usize),
            count: Vec::with_capacity(output as usize),
            weights: vec![0; output as usize * ksize],
        };
        let mut taps = vec![0f64; ksize];
        for dst in 0..output as usize {
            let center = (dst as f64 + 0.5) * scale;
            let first = ((center - support + 0.5) as i64).max(0) as usize;
            let last = ((center + support + 0.5) as i64).min(i64::from(input)) as usize;
            let count = last.saturating_sub(first);
            let mut sum = 0.0;
            for (k, tap) in taps.iter_mut().enumerate().take(count) {
                *tap = bicubic(((first + k) as f64 - center + 0.5) / filter_scale);
                sum += *tap;
            }
            for (k, &tap) in taps.iter().enumerate() {
                let normalized = if k < count && sum != 0.0 {
                    tap / sum
                } else {
                    0.0
                };
                // C rounds q ± 0.5 again; Pillow would truncate q + 0.5.
                let fixed = normalized * fixed_scale + if normalized < 0.0 { -0.5 } else { 0.5 };
                axis.weights[dst * ksize + k] = fixed.round() as i32;
            }
            axis.first.push(first);
            axis.count.push(count);
        }
        axis
    }

    /// First input index and the weights of output `dst`'s taps.
    fn taps(&self, dst: usize) -> (usize, &[i32]) {
        let start = dst * self.ksize;
        (
            self.first[dst],
            &self.weights[start..start + self.count[dst]],
        )
    }
}

fn bicubic(x: f64) -> f64 {
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

/// EXIF orientation values 1..=8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orientation(u16);

impl Orientation {
    const NORMAL: Self = Self(1);

    fn transposes(self) -> bool {
        self.0 >= 5
    }
}

/// The JPEG APP1 Exif orientation tag (ds4_glm53_image.inc); anything
/// malformed or absent is normal.
pub fn jpeg_orientation(data: &[u8]) -> Orientation {
    const SOS: u8 = 0xda;
    const EOI: u8 = 0xd9;
    const APP1: u8 = 0xe1;
    const TAG: u16 = 0x0112;
    const SHORT: u16 = 3;
    const ENTRY: usize = 12;
    if data.len() < 4 || data[0] != 0xff || data[1] != 0xd8 {
        return Orientation::NORMAL;
    }
    let mut pos = 2;
    while pos < data.len() && data[pos] == 0xff {
        while pos < data.len() && data[pos] == 0xff {
            pos += 1;
        }
        let Some(&marker) = data.get(pos) else {
            break;
        };
        pos += 1;
        if marker == SOS || marker == EOI {
            break;
        }
        if marker == 0x01 || marker == 0xd8 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        if data.len() - pos < 2 {
            break;
        }
        let segment = usize::from(u16::from_be_bytes([data[pos], data[pos + 1]]));
        if segment < 2 || segment > data.len() - pos {
            break;
        }
        if marker != APP1 || segment < 16 || &data[pos + 2..pos + 8] != b"Exif\0\0" {
            pos += segment;
            continue;
        }
        let tiff = &data[pos + 8..pos + segment];
        let le = &tiff[..2] == b"II";
        if (!le && &tiff[..2] != b"MM") || read16(tiff, 2, le) != 42 {
            break;
        }
        let ifd = read32(tiff, 4, le) as usize;
        if ifd > tiff.len() || tiff.len() - ifd < 2 {
            break;
        }
        let count = usize::from(read16(tiff, ifd, le));
        if count > (tiff.len() - ifd - 2) / ENTRY {
            break;
        }
        for i in 0..count {
            let at = ifd + 2 + i * ENTRY;
            if read16(tiff, at, le) != TAG
                || read16(tiff, at + 2, le) != SHORT
                || read32(tiff, at + 4, le) != 1
            {
                continue;
            }
            let value = read16(tiff, at + 8, le);
            return if (1..=8).contains(&value) {
                Orientation(value)
            } else {
                Orientation::NORMAL
            };
        }
        pos += segment;
    }
    Orientation::NORMAL
}

fn read16(b: &[u8], at: usize, le: bool) -> u16 {
    let v = [b[at], b[at + 1]];
    if le {
        u16::from_le_bytes(v)
    } else {
        u16::from_be_bytes(v)
    }
}

fn read32(b: &[u8], at: usize, le: bool) -> u32 {
    let v = [b[at], b[at + 1], b[at + 2], b[at + 3]];
    if le {
        u32::from_le_bytes(v)
    } else {
        u32::from_be_bytes(v)
    }
}

/// Applies the orientation the way ds4_glm53_image.inc maps each source
/// pixel to its destination.
fn orient(rgb: Rgb, orientation: Orientation) -> Rgb {
    if orientation == Orientation::NORMAL {
        return rgb;
    }
    let (w, h) = (rgb.width, rgb.height);
    let (out_w, out_h) = if orientation.transposes() {
        (h, w)
    } else {
        (w, h)
    };
    let mut data = vec![0u8; rgb.data.len()];
    for y in 0..h {
        for x in 0..w {
            let (dx, dy) = match orientation.0 {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (h - 1 - y, x),
                7 => (h - 1 - y, w - 1 - x),
                8 => (y, w - 1 - x),
                _ => (x, y),
            };
            let src = ((y * w + x) * 3) as usize;
            let dst = ((dy * out_w + dx) * 3) as usize;
            data[dst..dst + 3].copy_from_slice(&rgb.data[src..src + 3]);
        }
    }
    Rgb {
        width: out_w,
        height: out_h,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    const FIXTURES: [&str; 9] = [
        "tests/fixtures/qwen-images/small.png",
        "tests/fixtures/qwen-images/large.png",
        "tests/fixtures/qwen-images/screen.png",
        "tests/fixtures/qwen-images/screen-changed.png",
        "tests/fixtures/vision/homer.png",
        "tests/fixtures/vision/review-market-panel.png",
        "tests/fixtures/vision/bug.jpeg",
        "tests/fixtures/glm53/exif6.jpg",
        "tests/fixtures/glm53/exif8.jpg",
    ];
    const TOWERS: [(Tower, &str); 3] = [
        (Tower::Qwen, "qwen"),
        (Tower::Ling, "ling"),
        (Tower::Glm, "glm"),
    ];

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// Runs tests/vit_image_ref (make tests/vit_image_ref); None on failure.
    fn reference(args: &[&str]) -> Option<String> {
        let out = Command::new(repo().join("tests/vit_image_ref"))
            .args(args)
            .output()
            .expect("build the reference first: make tests/vit_image_ref");
        out.status
            .success()
            .then(|| String::from_utf8(out.stdout).unwrap())
    }

    fn plan_line(p: &ImagePlan) -> String {
        format!(
            "{} {} {} {} {} {} {} {} {}",
            p.source_width,
            p.source_height,
            p.content_width,
            p.content_height,
            p.padded_width,
            p.padded_height,
            p.grid_h,
            p.grid_w,
            p.token_count
        )
    }

    fn floats(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// The C path C hosts still run, file by file: plans, Qwen hashes and
    /// packed rows. PNG must match bit for bit; JPEG decoders may differ by
    /// a level, which is reported.
    #[test]
    #[ignore]
    fn matches_c_reference() {
        let dir = std::env::temp_dir().join(format!("ds4-vit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut failures = Vec::new();
        for fixture in FIXTURES {
            let path = repo().join(fixture);
            let file = path.to_str().unwrap();
            let data = std::fs::read(&path).unwrap();
            let jpeg = fixture.ends_with(".jpg") || fixture.ends_with(".jpeg");

            let c_hash = reference(&["hash", file]).map(|s| s.trim().to_string());
            let rust_hash = decode(&data).map(|rgb| pixel_hash(&rgb).to_string()).ok();
            if c_hash != rust_hash && !jpeg {
                failures.push(format!("{fixture}: hash {c_hash:?} vs {rust_hash:?}"));
            }

            for (tower, name) in TOWERS {
                let out = dir.join(format!("{name}.bin"));
                let c = reference(&["pack", name, file, out.to_str().unwrap()]);
                let plan = probe(tower, &data);
                let (Some(c), Ok(plan)) = (c, plan.as_ref()) else {
                    if reference(&["probe", name, file]).is_some() != plan.is_ok() {
                        failures.push(format!("{fixture} {name}: probe outcome differs"));
                    }
                    continue;
                };
                if c.trim() != plan_line(plan) {
                    failures.push(format!(
                        "{fixture} {name}: plan {} vs {}",
                        c.trim(),
                        plan_line(plan)
                    ));
                    continue;
                }

                let start = Instant::now();
                let rust = pack(tower, &data, plan).unwrap();
                let elapsed = start.elapsed();
                let c = floats(&out);
                let (mut differ, mut max) = (0usize, 0f32);
                for (a, b) in rust.iter().zip(&c) {
                    if a.to_bits() != b.to_bits() {
                        differ += 1;
                        max = max.max((a - b).abs());
                    }
                }
                eprintln!(
                    "{fixture} {name}: {} rows, rust pack {:.1} ms, {differ} of {} floats differ (max {max:.3e})",
                    plan.grid_h * plan.grid_w,
                    elapsed.as_secs_f64() * 1e3,
                    c.len(),
                );
                // PNG pixels are exact; only GLM's normalization may move an ulp.
                let png_ok = match tower {
                    Tower::Glm => max <= 1e-6,
                    _ => differ == 0,
                };
                if rust.len() != c.len() || (!jpeg && !png_ok) {
                    failures.push(format!("{fixture} {name}: {differ} floats differ"));
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// `DS4_VIT_BENCH=qwen:FILE:N`: median ms per decode+resize+pack and
    /// the process peak RSS, the numbers `vit_image_ref bench` prints for C.
    #[test]
    #[ignore]
    fn bench() {
        let spec = std::env::var("DS4_VIT_BENCH").expect("DS4_VIT_BENCH=tower:file:n");
        let mut parts = spec.splitn(3, ':');
        let (name, file, n) = (
            parts.next().unwrap(),
            parts.next().unwrap(),
            parts.next().unwrap(),
        );
        let tower = TOWERS.iter().find(|(_, t)| *t == name).expect("tower").0;
        let data = std::fs::read(repo().join(file)).unwrap();
        let plan = probe(tower, &data).unwrap();
        let mut ms: Vec<f64> = (0..n.parse::<usize>().unwrap())
            .map(|_| {
                let start = Instant::now();
                std::hint::black_box(pack(tower, &data, &plan).unwrap());
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        let usage = unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            libc::getrusage(libc::RUSAGE_SELF, &mut usage);
            usage
        };
        eprintln!(
            "rust {name} median {:.2} ms min {:.2} ms peak_rss {} KiB",
            ms[ms.len() / 2],
            ms[0],
            usage.ru_maxrss
        );
    }

    /// `DS4_VIT_DECODE=IN:OUT` writes the decoded RGB bytes, to compare
    /// decoders (stb_image via vit_image_ref decode, PIL).
    #[test]
    #[ignore]
    fn decode_dump() {
        let spec = std::env::var("DS4_VIT_DECODE").expect("DS4_VIT_DECODE=in:out");
        let (input, output) = spec.split_once(':').unwrap();
        let rgb = decode(&std::fs::read(repo().join(input)).unwrap()).unwrap();
        std::fs::write(output, &rgb.data).unwrap();
    }

    /// JPEG pixels are PIL's: SHA-256 of `Image.open(f).convert("RGB")
    /// .tobytes()` under Pillow 12.1.1 (libjpeg-turbo 3.1.3).
    #[test]
    fn jpeg_matches_pil() {
        const PIL: [(&str, &str); 7] = [
            (
                "tests/fixtures/vit-jpeg/odd420.jpg",
                "b74cc002df970c12b4eaae511f16ee339d92e195b96e10795cc6e0a01922ec45",
            ),
            (
                "tests/fixtures/vit-jpeg/h2v1.jpg",
                "8994ed239fd357c157d8a58bca97bf06fdceda6de114d695471415cdd2bbe216",
            ),
            (
                "tests/fixtures/vit-jpeg/progressive.jpg",
                "5477708636ca9de50bed74be2e35729e38ef83a9654976d9f1c69ff6626976e6",
            ),
            (
                "tests/fixtures/vit-jpeg/gray.jpg",
                "9702bbe2bf1ebf7c9980af79694563ea36710e3ad3025f81505ea2701773495d",
            ),
            (
                "tests/fixtures/vision/bug.jpeg",
                "85ac4b5c098b679307161969c7edef78c798ce3cea1b28b4c51dff05a1b5af67",
            ),
            (
                "tests/fixtures/qwen-images/photo.jpg",
                "192061df33aa0101975c6a4000b321f6f2a2bba10aacd3a7601a652248d51ff0",
            ),
            (
                "tests/fixtures/glm53/exif6.jpg",
                "1d53f3b5ff8d24a16fc7c260e9cde92089b3b941ac743f5f5e0d1a755c7e3d55",
            ),
        ];
        for (file, want) in PIL {
            let rgb = decode(&std::fs::read(repo().join(file)).unwrap()).unwrap();
            let got: String = Sha256::digest(&rgb.data)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(got, want, "{file}");
        }
    }

    #[test]
    fn exif_orientation_reads_fixtures() {
        let read =
            |name: &str| std::fs::read(repo().join("tests/fixtures/glm53").join(name)).unwrap();
        assert_eq!(jpeg_orientation(&read("exif6.jpg")), Orientation(6));
        assert_eq!(jpeg_orientation(&read("exif8.jpg")), Orientation(8));
        assert_eq!(jpeg_orientation(b"\x89PNG"), Orientation::NORMAL);
    }

    #[test]
    fn plans_follow_the_processors() {
        // Qwen: 1000x700 rounds to 992x704 (factor 32) -> 62x44 grid.
        let q = plan_qwen(1000, 700, QWEN_MIN_PIXELS, SOURCE_MAX_PIXELS).unwrap();
        assert_eq!(
            (q.content_width, q.content_height, q.grid_w, q.grid_h),
            (992, 704, 62, 44)
        );
        assert_eq!(q.token_count, 62 * 44 / 4);
        // Ling caps at 4 MiP: the same image stays, a 4096x4096 one shrinks.
        let l = plan_qwen(4096, 4096, LING_MIN_PIXELS, LING_MAX_PIXELS).unwrap();
        assert!(u64::from(l.content_width) * u64::from(l.content_height) <= LING_MAX_PIXELS);
        assert!(plan_qwen(10, 3000, QWEN_MIN_PIXELS, SOURCE_MAX_PIXELS).is_err());
    }
}
