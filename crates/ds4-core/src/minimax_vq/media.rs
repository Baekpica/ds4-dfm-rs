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
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

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
fn pack(frames: &[&[u8]], height: u32, width: u32) -> Visual {
    let mean = MEAN.map(|m| m * 255.0);
    let std = STD.map(|s| s * 255.0);
    let (grid_h, grid_w) = (height / PATCH, width / PATCH);
    let grid_t = frames.len() as u32 / TEMPORAL;
    let mut patches = Vec::with_capacity((grid_t * grid_h * grid_w) as usize * PATCH_VALUES);
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
    Visual {
        patches,
        grid_t,
        grid_h,
        grid_w,
    }
}

/// A PNG or JPEG still: decoded, resized, one frame doubled to a pair.
pub fn image(data: &[u8]) -> Result<Visual, String> {
    let rgb = image::load_from_memory(data)
        .map_err(|e| format!("image decode failed: {e}"))?
        .into_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    let (th, tw) = smart_resize(height, width, IMAGE_PIXELS)?;
    let pixels = if (th, tw) == (height, width) {
        rgb.into_raw()
    } else {
        pil_resize_rgb(rgb.as_raw(), width, height, tw, th)
    };
    Ok(pack(&[&pixels, &pixels], th, tw))
}

/// An MP4 clip: frames at VIDEO_FPS through ffmpeg (bicubic scaling to the
/// smart_resize size), an odd count repeating its last frame.
pub fn video(data: &[u8]) -> Result<Visual, String> {
    let path = std::env::temp_dir().join(format!(
        "ds4-minimax-{}-{}.mp4",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&path, data).map_err(|e| format!("video temp file: {e}"))?;
    let result = decode_video(&path);
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
    Ok(pack(&frames, height, width))
}

fn decode_video(path: &std::path::Path) -> Result<(Vec<u8>, u32, u32), String> {
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
    let width = meta["streams"][0]["width"].as_u64().unwrap_or(0) as u32;
    let height = meta["streams"][0]["height"].as_u64().unwrap_or(0) as u32;
    let (th, tw) = smart_resize(height, width, VIDEO_PIXELS)?;
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
    let mut raw = Vec::new();
    let mut stdout = child.stdout.take().ok_or("ffmpeg has no output")?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n =
            std::io::Read::read(&mut stdout, &mut buf).map_err(|e| format!("ffmpeg read: {e}"))?;
        if n == 0 {
            break;
        }
        if raw.len() + n > VIDEO_RAW_CAP {
            let _ = child.kill();
            let _ = child.wait();
            return Err("video decodes past the raw frame budget".into());
        }
        raw.extend_from_slice(&buf[..n]);
    }
    drop(stdout);
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
