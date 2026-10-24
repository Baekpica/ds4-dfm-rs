//! MiniMax-M3.1 image and video input as the shipped processor builds it
//! (image_processor.py, video_processor.py, processing_minimax.py; the
//! handoff's minimax_vision_ref.py is the contract): smart_resize to
//! multiples of 28, PIL's antialiased bicubic (the fast processor's
//! torchvision resize differs by at most 2/255 on ~0.1% of the bytes), CLIP
//! mean/std, patches in the processor's order, a still image doubled into one
//! temporal pair. A prompt span is the vision start token, one pad per
//! merged 2x2 group of patches and the end token; a video writes each frame
//! pair's timestamp before its span.

use crate::v41_media::pil_resize_rgb;

pub const PATCH: u32 = 14;
pub const MERGE: u32 = 2;
pub const TEMPORAL: u32 = 2;
const FACTOR: u32 = PATCH * MERGE;
const IMAGE_PIXELS: (u32, u32) = (3_136, 451_584);
const VIDEO_PIXELS: (u32, u32) = (3_136, 602_112);
const MAX_RATIO: f64 = 200.0;
/// Values per patch: channel, then temporal frame, then 14 x 14 pixels.
pub const PATCH_VALUES: usize = 3 * (TEMPORAL * PATCH * PATCH) as usize;
/// Temporal pairs per attention segment of a video (vision_segment_max_frames).
pub const SEGMENT_PAIRS: u32 = 4;
pub const IMAGE_PAD: i32 = 200_025;
pub const VIDEO_PAD: i32 = 200_026;
pub const VISION_START: i32 = 200_029;
pub const VISION_END: i32 = 200_030;
/// Videos are sampled at 1 frame per second, at most 768 frames.
pub const VIDEO_FPS: u32 = 1;
const VIDEO_MAX_FRAMES: usize = 768;
/// Raw RGB a video may decode to before it is refused.
const VIDEO_RAW_CAP: usize = 512 * 1024 * 1024;
const VIDEO_READ_BYTES: usize = 1 << 20;
const VISION_WIDTH: u64 = 6_144;
const VISION_DIM: u64 = 1_280;
const VISION_FF: u64 = 5_120;
const VISION_PASS_ROWS: u64 = 12_288;
const VISION_SCORE_BYTES: u64 = 192 << 20;
const VISION_CELL_BYTES: u64 = 96;
const VISION_ROW_BYTES: u64 = 61_440;
const VISION_STAGE_BYTES: u64 = (8 << 20) + 4 * VISION_WIDTH * 4;
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

#[derive(Debug)]
pub enum Error {
    Capacity,
    Invalid(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capacity => f.write_str("MiniMax media exceeds live memory capacity; retry later"),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::Invalid(message.into())
    }
}

#[derive(Clone, Copy)]
struct MediaCost {
    patches: u64,
    outputs: u64,
    transient: u64,
}

/// New media allocations after the caller reserves its pending graph and
/// memory floor. Patches persist together; encoders run one visual at a time.
pub struct MediaBudget {
    limit: u64,
    patches: u64,
    outputs: u64,
    transient: u64,
}

impl MediaBudget {
    pub fn new(bytes: u64) -> Self {
        Self {
            limit: bytes,
            patches: 0,
            outputs: 0,
            transient: 0,
        }
    }

    /// After preparation, patches are already charged to live availability.
    pub fn reserve_bytes(&self) -> u64 {
        self.outputs.saturating_add(self.transient)
    }

    fn room(&self) -> u64 {
        self.limit.saturating_sub(self.patches)
    }

    fn check(&self, cost: MediaCost, preprocessing: u64) -> Result<(), Error> {
        let retained = self.patches.saturating_add(cost.patches);
        let sync = retained
            .saturating_add(self.outputs)
            .saturating_add(cost.outputs)
            .saturating_add(self.transient.max(cost.transient));
        let prepare = self.patches.saturating_add(preprocessing);
        if sync.max(prepare) > self.limit {
            return Err(Error::Capacity);
        }
        Ok(())
    }

    fn retain(&mut self, cost: MediaCost) {
        self.patches += cost.patches;
        self.outputs += cost.outputs;
        self.transient = self.transient.max(cost.transient);
    }
}

fn media_cost(pairs: u32, height: u32, width: u32) -> Result<MediaCost, Error> {
    let per = u64::from(height / PATCH) * u64::from(width / PATCH);
    if pairs == 0 || per == 0 || per > VISION_PASS_ROWS {
        return Err("invalid MiniMax media geometry".into());
    }
    let segment = u64::from(SEGMENT_PAIRS);
    let mut pass_pairs = VISION_PASS_ROWS / per / segment * segment;
    if pass_pairs == 0 {
        pass_pairs = if segment <= VISION_PASS_ROWS / per { segment } else { 1 };
    }
    if pass_pairs < segment && u64::from(pairs) > pass_pairs {
        return Err("MiniMax video segment exceeds one vision pass".into());
    }
    let n = u64::from(pairs).min(pass_pairs) * per;
    let m = u64::from(pairs).min(segment) * per;
    let q = m.min(VISION_SCORE_BYTES / (VISION_CELL_BYTES * m));
    // ds4_minimax_vision.inc/mm_vis_pass and the cleared native scratch slots:
    // tower and projector peaks are disjoint; clean streamed weights are reclaimable.
    let tower_fixed = 2 * VISION_DIM * (4 * VISION_DIM + 2 * VISION_FF)
        + 4 * (3 * VISION_DIM + VISION_FF);
    let tower = tower_fixed + 6 * VISION_DIM * m + VISION_CELL_BYTES * m * q;
    let projector = 8 * VISION_WIDTH * VISION_WIDTH + 4 * VISION_WIDTH;
    let transient = VISION_ROW_BYTES * n + tower.max(projector) + VISION_STAGE_BYTES;
    let patches = per * u64::from(pairs);
    Ok(MediaCost {
        patches: patches * PATCH_VALUES as u64 * 4,
        // Four patches merge; their F32 byte width cancels that divisor.
        outputs: patches * VISION_WIDTH,
        transient,
    })
}

fn image_error(error: image::ImageError) -> Error {
    match error {
        image::ImageError::Limits(_) => Error::Capacity,
        error => Error::Invalid(format!("image decode failed: {error}")),
    }
}

fn coeff_bytes(input: u32, output: u32) -> u64 {
    let support = (2.0 * (f64::from(input) / f64::from(output)).max(1.0)).ceil() as u64;
    // pil_coeffs holds F64 and I32 coefficients together during conversion.
    u64::from(output).saturating_mul((2 * support + 1).saturating_mul(12).saturating_add(8))
}

fn image_work(decoded: u64, height: u32, width: u32, th: u32, tw: u32, patches: u64) -> u64 {
    let source = u64::from(height)
        .saturating_mul(u64::from(width))
        .saturating_mul(3);
    let target = u64::from(th) * u64::from(tw) * 3;
    let horizontal = u64::from(height)
        .saturating_mul(u64::from(tw))
        .saturating_mul(3);
    let mut peak = decoded
        .saturating_add(source)
        .max(target.saturating_add(patches));
    if (th, tw) != (height, width) {
        let horizontal_peak = if tw != width {
            source.saturating_mul(2)
                .saturating_add(horizontal)
                .saturating_add(coeff_bytes(width, tw))
        } else {
            source.saturating_mul(2)
        };
        let vertical_peak = source
            .saturating_add(horizontal)
            .saturating_add(target)
            .saturating_add(if th != height { coeff_bytes(height, th) } else { 0 });
        peak = peak.max(horizontal_peak).max(vertical_peak);
    }
    peak
}

fn video_growth(
    budget: &MediaBudget,
    bytes: usize,
    capacity: usize,
    old_capacity: usize,
    height: u32,
    width: u32,
) -> Result<(), Error> {
    let frame = height as usize * width as usize * 3;
    let pairs = bytes.max(1).div_ceil(frame).div_ceil(TEMPORAL as usize) as u32;
    let cost = media_cost(pairs, height, width)?;
    let references = (VIDEO_MAX_FRAMES + 1) * std::mem::size_of::<&[u8]>();
    let work = (capacity as u64)
        .saturating_add((old_capacity as u64).max(cost.patches))
        .saturating_add((VIDEO_READ_BYTES + references) as u64);
    budget.check(cost, work)
}

/// Patches of one image or video in the processor's order.
#[derive(Clone, Debug)]
pub struct Visual {
    /// [grid_t * grid_h * grid_w][PATCH_VALUES] F32.
    pub patches: Vec<f32>,
    pub grid_t: u32,
    pub grid_h: u32,
    pub grid_w: u32,
}

impl Visual {
    /// Language rows of one frame pair (2 x 2 patches merge into one).
    pub fn pair_tokens(&self) -> u32 {
        self.grid_h * self.grid_w / (MERGE * MERGE)
    }

    pub fn tokens(&self) -> u32 {
        self.grid_t * self.pair_tokens()
    }

    /// Identity of the encoded input for cache reuse: geometry and patches.
    pub fn tag(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut feed = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
        };
        for dim in [self.grid_t, self.grid_h, self.grid_w] {
            feed(&dim.to_le_bytes());
        }
        for value in &self.patches {
            feed(&value.to_bits().to_le_bytes());
        }
        hash
    }
}

fn round_by(n: f64) -> u32 {
    ((n / f64::from(FACTOR)).round_ties_even() as u32) * FACTOR
}

/// image_processor.py `smart_resize` (video_processor.py's is the same).
pub fn smart_resize(
    height: u32,
    width: u32,
    (min_pixels, max_pixels): (u32, u32),
) -> Result<(u32, u32), String> {
    if height == 0 || width == 0 {
        return Err("image has zero extent".into());
    }
    let (h, w) = (f64::from(height), f64::from(width));
    if h.max(w) / h.min(w) > MAX_RATIO {
        return Err(format!(
            "image aspect ratio exceeds {MAX_RATIO}: {height}x{width}"
        ));
    }
    let mut h_bar = round_by(h).max(FACTOR);
    let mut w_bar = round_by(w).max(FACTOR);
    let factor = f64::from(FACTOR);
    if u64::from(h_bar) * u64::from(w_bar) > u64::from(max_pixels) {
        let beta = (h * w / f64::from(max_pixels)).sqrt();
        h_bar = ((h / beta / factor).floor() as u32) * FACTOR;
        w_bar = ((w / beta / factor).floor() as u32) * FACTOR;
    } else if u64::from(h_bar) * u64::from(w_bar) < u64::from(min_pixels) {
        let beta = (f64::from(min_pixels) / (h * w)).sqrt();
        h_bar = ((h * beta / factor).ceil() as u32) * FACTOR;
        w_bar = ((w * beta / factor).ceil() as u32) * FACTOR;
    }
    if h_bar == 0 || w_bar == 0 {
        return Err("image resizes to zero extent".into());
    }
    Ok((h_bar, w_bar))
}

/// Normalized patches of an even number of RGB frames at (height, width):
/// per frame pair, 2 x 2 blocks of patches row-major, each block's four
/// patches row-major, each patch channel, frame, then 14 x 14 pixels.
fn pack(frames: &[&[u8]], height: u32, width: u32) -> Result<Visual, Error> {
    let mean = MEAN.map(|m| m * 255.0);
    let std = STD.map(|s| s * 255.0);
    let (grid_h, grid_w) = (height / PATCH, width / PATCH);
    let grid_t = frames.len() as u32 / TEMPORAL;
    let mut patches = Vec::new();
    patches
        .try_reserve_exact((grid_t * grid_h * grid_w) as usize * PATCH_VALUES)
        .map_err(|_| Error::Capacity)?;
    for pair in frames.chunks(TEMPORAL as usize) {
        for by in 0..grid_h / MERGE {
            for bx in 0..grid_w / MERGE {
                for dy in 0..MERGE {
                    for dx in 0..MERGE {
                        let (y0, x0) = ((by * MERGE + dy) * PATCH, (bx * MERGE + dx) * PATCH);
                        for channel in 0..3 {
                            for frame in pair {
                                for py in 0..PATCH {
                                    for px in 0..PATCH {
                                        let at =
                                            (((y0 + py) * width + x0 + px) * 3) as usize + channel;
                                        patches.push(
                                            (f32::from(frame[at]) - mean[channel]) / std[channel],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Visual {
        patches,
        grid_t,
        grid_h,
        grid_w,
    })
}

/// A PNG or JPEG still: decoded, resized, one frame doubled to a pair.
pub fn image(data: &[u8]) -> Result<Visual, String> {
    image_budget(data, &mut MediaBudget::new(u64::MAX)).map_err(|e| e.to_string())
}

/// Check dimensions and decoder allocations before materializing source pixels.
pub fn image_budget(data: &[u8], budget: &mut MediaBudget) -> Result<Visual, Error> {
    use image::ImageDecoder;

    if budget.room() == 0 {
        return Err(Error::Capacity);
    }
    let mut reader = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| Error::Invalid(format!("image header failed: {e}")))?;
    // image 0.25's JPEG decoder owns a copy of its compressed input.
    let decoder_input = if reader.format() == Some(image::ImageFormat::Jpeg) {
        data.len() as u64
    } else {
        0
    };
    let decode_room = budget.room().checked_sub(decoder_input).ok_or(Error::Capacity)?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(limits.max_alloc.unwrap_or(u64::MAX).min(decode_room));
    let edge = decode_room.min(u64::from(u32::MAX)) as u32;
    limits.max_image_width = Some(edge);
    limits.max_image_height = Some(edge);
    reader.limits(limits.clone());
    let mut decoder = reader.into_decoder().map_err(image_error)?;
    let (width, height) = decoder.dimensions();
    let (th, tw) = smart_resize(height, width, IMAGE_PIXELS)?;
    let cost = media_cost(1, th, tw)?;
    let work = image_work(decoder.total_bytes(), height, width, th, tw, cost.patches)
        .saturating_add(decoder_input);
    budget.check(cost, work)?;
    // Match ImageReader::decode: its output buffer also consumes max_alloc.
    limits.reserve(decoder.total_bytes()).map_err(image_error)?;
    decoder.set_limits(limits).map_err(image_error)?;
    let rgb = image::DynamicImage::from_decoder(decoder)
        .map_err(image_error)?
        .into_rgb8();
    let pixels = if (th, tw) == (height, width) {
        rgb.into_raw()
    } else {
        let pixels = pil_resize_rgb(rgb.as_raw(), width, height, tw, th);
        drop(rgb);
        pixels
    };
    let visual = pack(&[&pixels, &pixels], th, tw)?;
    budget.retain(cost);
    Ok(visual)
}

/// An MP4 clip: frames at VIDEO_FPS through ffmpeg (bicubic scaling to the
/// smart_resize size), an odd count repeating its last frame.
pub fn video(data: &[u8]) -> Result<Visual, String> {
    video_budget(data, &mut MediaBudget::new(u64::MAX)).map_err(|e| e.to_string())
}

/// Decode incrementally within the request's remaining memory allowance.
pub fn video_budget(data: &[u8], budget: &mut MediaBudget) -> Result<Visual, Error> {
    if budget.room() == 0 || data.len() as u64 > budget.room() {
        return Err(Error::Capacity);
    }
    let path = std::env::temp_dir().join(format!(
        "ds4-minimax-{}-{}.mp4",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let result = (|| -> Result<_, Error> {
        let mut file = std::fs::File::create(&path).map_err(|e| format!("video temp file: {e}"))?;
        std::io::Write::write_all(&mut file, data).map_err(|e| format!("video temp file: {e}"))?;
        // The source file becomes clean/reclaimable before raw frames grow.
        file.sync_data().map_err(|e| format!("video temp file writeback: {e}"))?;
        drop(file);
        decode_video(&path, budget)
    })();
    let _ = std::fs::remove_file(&path);
    let (raw, height, width) = result?;
    let frame = height as usize * width as usize * 3;
    if raw.is_empty() || raw.len() % frame != 0 {
        return Err("video produced no whole frames".into());
    }
    let mut frames: Vec<&[u8]> = raw.chunks(frame).take(VIDEO_MAX_FRAMES).collect();
    if frames.len() % 2 == 1 {
        frames.push(frames[frames.len() - 1]);
    }
    let cost = media_cost(frames.len() as u32 / TEMPORAL, height, width)?;
    budget.check(cost, (raw.capacity() as u64).saturating_add(cost.patches))?;
    let visual = pack(&frames, height, width)?;
    budget.retain(cost);
    Ok(visual)
}

fn decode_video(path: &std::path::Path, budget: &MediaBudget) -> Result<(Vec<u8>, u32, u32), Error> {
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|e| format!("ffprobe failed to start: {e}"))?;
    if !probe.status.success() {
        return Err("ffprobe rejected the video".into());
    }
    let meta: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap_or_default();
    let width = meta["streams"][0]["width"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(0);
    let height = meta["streams"][0]["height"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(0);
    let (th, tw) = smart_resize(height, width, VIDEO_PIXELS)?;
    video_growth(budget, 1, 0, 0, th, tw)?;
    let mut child = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("fps={VIDEO_FPS},scale={tw}:{th}:flags=bicubic+accurate_rnd+full_chroma_int"),
            "-frames:v",
            &VIDEO_MAX_FRAMES.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("ffmpeg failed to start: {e}"))?;
    let decoded = (|| -> Result<Vec<u8>, Error> {
        let mut raw = Vec::new();
        let mut stdout = child.stdout.take().ok_or("ffmpeg has no output")?;
        let mut buf = vec![0u8; VIDEO_READ_BYTES];
        loop {
            let n = std::io::Read::read(&mut stdout, &mut buf)
                .map_err(|e| format!("ffmpeg read: {e}"))?;
            if n == 0 {
                break;
            }
            let bytes = raw.len() + n;
            if bytes > VIDEO_RAW_CAP {
                return Err("video decodes past the raw frame budget".into());
            }
            let old_capacity = raw.capacity();
            let capacity = if bytes > old_capacity {
                old_capacity.saturating_mul(2).max(bytes).min(VIDEO_RAW_CAP)
            } else {
                old_capacity
            };
            let growing = if capacity > old_capacity { old_capacity } else { 0 };
            video_growth(budget, bytes, capacity, growing, th, tw)?;
            if capacity > old_capacity {
                raw.try_reserve_exact(capacity - raw.len())
                    .map_err(|_| Error::Capacity)?;
            }
            raw.extend_from_slice(&buf[..n]);
        }
        Ok(raw)
    })();
    let raw = match decoded {
        Ok(raw) => raw,
        Err(error) => {
            // Capacity and read failures must not leave a decoder running.
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    };
    if !child.wait().map(|s| s.success()).unwrap_or(false) {
        return Err("ffmpeg failed to decode the video".into());
    }
    Ok((raw, th, tw))
}

/// The span replacing one image placeholder.
pub fn image_span(visual: &Visual) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(visual.tokens() as usize + 2);
    tokens.push(VISION_START);
    tokens.extend(std::iter::repeat_n(IMAGE_PAD, visual.tokens() as usize));
    tokens.push(VISION_END);
    tokens
}

/// Timestamp text written before frame pair `pair` (first frame index / fps).
pub fn timestamp(pair: u32) -> String {
    format!(
        "]<]{:.1} seconds[>[",
        (pair * TEMPORAL) as f32 / VIDEO_FPS as f32
    )
}

/// One run of pads the encoder's rows replace: prompt position, count, and
/// which visual's rows (from `row`) fill it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub count: u32,
    pub visual: usize,
    pub row: u32,
}

/// Replace each image (200025) and video (200026) placeholder of `tokens`,
/// in order, by its spans; `timestamp_ids(pair)` tokenizes a video pair's
/// timestamp text.
pub fn expand(
    tokens: &[i32],
    visuals: &[(bool, &Visual)],
    mut timestamp_ids: impl FnMut(u32) -> Result<Vec<i32>, String>,
) -> Result<(Vec<i32>, Vec<Span>), String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut spans = Vec::new();
    let mut next = 0usize;
    for &token in tokens {
        if token != IMAGE_PAD && token != VIDEO_PAD {
            out.push(token);
            continue;
        }
        let (is_video, visual) = *visuals
            .get(next)
            .ok_or("more media placeholders than inputs")?;
        if is_video != (token == VIDEO_PAD) {
            return Err("media placeholder kind does not match its input".into());
        }
        let per = visual.pair_tokens();
        let pairs = if is_video { visual.grid_t } else { 1 };
        for pair in 0..pairs {
            if is_video {
                out.extend(timestamp_ids(pair)?);
            }
            out.push(VISION_START);
            spans.push(Span {
                start: out.len() as u32,
                count: per,
                visual: next,
                row: pair * per,
            });
            out.extend(std::iter::repeat_n(token, per as usize));
            out.push(VISION_END);
        }
        next += 1;
    }
    if next != visuals.len() {
        return Err("media inputs without a placeholder".into());
    }
    Ok((out, spans))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_budget_matches_passes() {
        assert_eq!(media_cost(4, 672, 896).unwrap().transient, 1_097_731_072);
        assert_eq!(media_cost(1, 672, 672).unwrap().transient, 452_059_136);
    }

    #[test]
    fn media_budget_counts_retained() {
        let cost = media_cost(1, 56, 56).unwrap();
        let total = 2 * (cost.patches + cost.outputs) + cost.transient;
        let mut budget = MediaBudget::new(total);
        budget.check(cost, 0).unwrap();
        budget.retain(cost);
        budget.check(cost, 0).unwrap();
        budget.retain(cost);
        assert_eq!(budget.reserve_bytes(), 2 * cost.outputs + cost.transient);
        assert!(matches!(budget.check(cost, 0), Err(Error::Capacity)));

        let mut short = MediaBudget::new(total - 1);
        short.retain(cost);
        assert!(matches!(short.check(cost, 0), Err(Error::Capacity)));
    }

    #[test]
    fn video_budget_prices_padded_pair() {
        let frame = 56 * 56 * 3;
        let one = media_cost(1, 56, 56).unwrap();
        let budget = MediaBudget::new(one.patches + one.outputs + one.transient);
        video_growth(&budget, 2 * frame, 2 * frame, 0, 56, 56).unwrap();
        assert!(matches!(
            video_growth(&budget, 2 * frame + 1, 3 * frame, 0, 56, 56),
            Err(Error::Capacity)
        ));
    }

    #[test]
    fn image_budget_refuses_before_pack() {
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(image::RgbImage::new(1, 1))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let mut budget = MediaBudget::new(0);
        assert!(matches!(image_budget(&png, &mut budget), Err(Error::Capacity)));
        assert_eq!(budget.patches, 0);
        assert_eq!(budget.reserve_bytes(), 0);
    }

    #[test]
    fn resize_matches_the_processor() {
        // minimax_vision_ref.py on a 500 x 731 image: 504 x 728, grid 36 x 52.
        assert_eq!(smart_resize(500, 731, IMAGE_PIXELS), Ok((504, 728)));
        assert_eq!(smart_resize(10, 10, IMAGE_PIXELS), Ok((56, 56)));
        assert!(smart_resize(1, 300, IMAGE_PIXELS).is_err());
        let (h, w) = smart_resize(4000, 3000, IMAGE_PIXELS).unwrap();
        assert!(u64::from(h) * u64::from(w) <= 451_584 && h % 28 == 0 && w % 28 == 0);
    }

    #[test]
    fn spans_follow_placeholders() {
        let still = Visual {
            patches: Vec::new(),
            grid_t: 1,
            grid_h: 4,
            grid_w: 6,
        };
        let clip = Visual {
            patches: Vec::new(),
            grid_t: 2,
            grid_h: 2,
            grid_w: 2,
        };
        let (tokens, spans) = expand(
            &[7, IMAGE_PAD, 8, VIDEO_PAD],
            &[(false, &still), (true, &clip)],
            |p| Ok(vec![900 + p as i32]),
        )
        .unwrap();
        let mut want = vec![7, VISION_START];
        want.extend([IMAGE_PAD; 6]);
        want.extend([
            VISION_END,
            8,
            900,
            VISION_START,
            VIDEO_PAD,
            VISION_END,
            901,
            VISION_START,
            VIDEO_PAD,
            VISION_END,
        ]);
        assert_eq!(tokens, want);
        assert_eq!(
            spans,
            vec![
                Span {
                    start: 2,
                    count: 6,
                    visual: 0,
                    row: 0
                },
                Span {
                    start: 12,
                    count: 1,
                    visual: 1,
                    row: 0
                },
                Span {
                    start: 16,
                    count: 1,
                    visual: 1,
                    row: 1
                },
            ]
        );
        assert_eq!(timestamp(1), "]<]2.0 seconds[>[");
    }
}
