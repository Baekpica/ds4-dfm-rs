//! The release's processor supplies row-major RGB patches and one learned newline per merged row.
use image::{imageops, ImageReader, RgbImage};
use std::io::Cursor;

pub const IMAGE: i32 = 120120;
pub const VIDEO: i32 = 120124;
pub const MAX_INPUTS: usize = 4;
const VIDEO_ROWS: u32 = 60_000;

// ds4_hy4_vision.inc stages one frame and one layer. RGB and the projected
// language rows of every input stay live until the prompt consumes them.
pub(crate) fn workspace_bytes(ctx: u64) -> u64 {
    let rows = ctx.min(u64::from(VIDEO_ROWS) * MAX_INPUTS as u64);
    let patches = (rows * 4).min(16_384);
    let tower_row = (5 * 1152 + 4304) * 4 + (3 * 1152 + 4304) * 2;
    let scores = (patches * patches * 16 * 6).min(192 << 20);
    let stage = 4608 * 4608 * 2 + (8 << 20);
    rows * (3072 * 4 + 32 * 32 * 3) + patches * (tower_row + 768 * 4) + scores + stage
}

#[derive(Debug)]
pub struct Visual {
    width: u32,
    height: u32,
    frames: u32,
    rgb: Vec<u8>,
    token: i32,
}

fn validate_size(w: u32, h: u32) -> Result<(), String> {
    if w == 0
        || h == 0
        || w.max(h) as u64 > 200 * w.min(h) as u64
        || w as u64 * h as u64 > 16_777_216
    {
        return Err("Hy4 visual dimensions exceed the processor limit".into());
    }
    Ok(())
}

fn image_size(w: u32, h: u32) -> Result<(u32, u32), String> {
    validate_size(w, h)?;
    let (w, h) = (w as f64, h as f64);
    let mut cols = (w / 32.).round_ties_even();
    let mut rows = (h / 32.).round_ties_even();
    let pixels = cols * rows * 1024.;
    if pixels > 4_194_304. {
        let scale = (4_194_304. / (w * h)).sqrt();
        cols = (w * scale / 32.).floor().max(1.);
        rows = (h * scale / 32.).floor().max(1.);
    } else if pixels < 4096. {
        let scale = (4096. / (w * h)).sqrt();
        cols = (w * scale / 32.).ceil();
        rows = (h * scale / 32.).ceil();
    }
    Ok((cols as u32 * 32, rows as u32 * 32))
}

fn video_size(w: u32, h: u32, frames: u32) -> Result<(u32, u32), String> {
    validate_size(w, h)?;
    if !(1..=2048).contains(&frames) {
        return Err("Hy4 MP4 frame count outside 1..2048".into());
    }
    for rows in (1..=(h / 32).max(1)).rev() {
        let lo = if rows == 1 { 1 } else { (w * rows / h).max(1) };
        let hi = (w * (rows + 1))
            .div_ceil(h)
            .saturating_sub(1)
            .max(1)
            .min((w / 32).max(1));
        let budget = (VIDEO_ROWS / (frames * rows)).saturating_sub(1);
        let cols = hi.min(budget).min(1024 / rows);
        if cols >= lo {
            return Ok((cols * 32, rows * 32));
        }
    }
    Ok((32, 32))
}

pub fn still(data: &[u8]) -> Result<Visual, String> {
    let (w, h) = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .into_dimensions()
        .map_err(|e| e.to_string())?;
    let (tw, th) = image_size(w, h)?;
    let rgb = crate::vit_image::decode(data)?;
    let img = RgbImage::from_raw(rgb.width, rgb.height, rgb.data).ok_or("invalid Hy4 image")?;
    let (tw, th) = if img.dimensions() == (w, h) {
        (tw, th)
    } else {
        image_size(img.width(), img.height())?
    };
    let rgb = if img.dimensions() == (tw, th) {
        img
    } else {
        imageops::resize(&img, tw, th, imageops::FilterType::Lanczos3)
    };
    Ok(Visual {
        width: tw,
        height: th,
        frames: 1,
        rgb: rgb.into_raw(),
        token: IMAGE,
    })
}

#[cfg(target_os = "linux")]
pub fn video(data: &[u8]) -> Result<Visual, String> {
    use std::io::{Read, Seek, Write};
    use std::os::fd::FromRawFd;
    use std::process::{Command, Stdio};
    // A seekable descriptor reads both front- and end-indexed MP4s. Neither decoder opens a request path.
    let fd = unsafe { libc::memfd_create(c"ds4-hy4-video".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    let mut input = unsafe { std::fs::File::from_raw_fd(fd) };
    input
        .write_all(data)
        .and_then(|_| input.rewind())
        .map_err(|e| e.to_string())?;
    let args = [
        "-v",
        "error",
        "-protocol_whitelist",
        "fd",
        "-f",
        "mov",
        "-enable_drefs",
        "0",
    ];
    let meta = Command::new("ffprobe")
        .args(args)
        .args([
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "json",
            "fd:",
        ])
        .stdin(input.try_clone().map_err(|e| e.to_string())?)
        .output()
        .map_err(|e| e.to_string())?;
    if !meta.status.success() {
        return Err("could not probe Hy4 MP4 video".into());
    }
    let meta: serde_json::Value =
        serde_json::from_slice(&meta.stdout).map_err(|e| e.to_string())?;
    let s = &meta["streams"][0];
    let dim = |k: &str| {
        s[k].as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0)
    };
    let frames = s["nb_frames"]
        .as_str()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let (w, h) = video_size(dim("width"), dim("height"), frames)?;
    input.rewind().map_err(|e| e.to_string())?;
    let mut child = Command::new("ffmpeg")
        .args(args)
        .args([
            "-noautorotate",
            "-threads",
            "1",
            "-i",
            "fd:",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &format!("scale={w}:{h}:flags=lanczos"),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ])
        .stdin(input)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    let bytes = w as usize * h as usize * frames as usize * 3;
    let mut rgb = Vec::new();
    let read = child
        .stdout
        .take()
        .unwrap()
        .take(bytes as u64 + 1)
        .read_to_end(&mut rgb);
    if read.is_err() || rgb.len() != bytes {
        let _ = child.kill();
        let _ = child.wait();
        return Err("invalid Hy4 MP4 frame count or geometry".into());
    }
    if !child.wait().map_err(|e| e.to_string())?.success() {
        return Err("Hy4 MP4 decode failed".into());
    }
    Ok(Visual {
        width: w,
        height: h,
        frames,
        rgb,
        token: VIDEO,
    })
}

#[cfg(not(target_os = "linux"))]
pub fn video(_: &[u8]) -> Result<Visual, String> {
    Err("Hy4 MP4 decoding requires Linux".into())
}

impl Visual {
    pub fn token(&self) -> i32 {
        self.token
    }
    pub fn tokens(&self) -> u32 {
        self.frames * (self.height / 32) * (self.width / 32 + 1)
    }

    fn patches(&self, frame: &[u8]) -> Vec<f32> {
        let w = self.width as usize;
        let mut v = Vec::with_capacity(frame.len());
        for y in (0..self.height as usize).step_by(16) {
            for x in (0..w).step_by(16) {
                for c in 0..3 {
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let b = frame[((y + dy) * w + x + dx) * 3 + c];
                            v.push((b as f32 / 255. - 0.5) * 2.);
                        }
                    }
                }
            }
        }
        v
    }
}

impl crate::Session<'_> {
    pub fn sync_hy4_media(
        &mut self,
        tokens: &crate::TokenBuffer,
        visuals: &[(u32, &Visual)],
    ) -> crate::Result<()> {
        use sha2::{Digest, Sha256};
        self.check_sync(tokens)?;
        let ids = tokens.as_slice();
        let bad = |s: &str| crate::Error {
            code: 1,
            message: s.into(),
        };
        if !(1..=MAX_INPUTS).contains(&visuals.len()) || ids.len() > i32::MAX as usize {
            return Err(bad("invalid Hy4 media prompt"));
        }
        let mut hash = Sha256::new();
        let mut spans = Vec::with_capacity(visuals.len());
        let mut features = Vec::with_capacity(visuals.len());
        let mut last = 0;
        for &(start, v) in visuals {
            let end = start
                .checked_add(v.tokens())
                .ok_or_else(|| bad("Hy4 media overflow"))?;
            if start < last
                || ids
                    .get(start as usize..end as usize)
                    .is_none_or(|s| !s.iter().all(|&id| id == v.token))
            {
                return Err(bad("Hy4 placeholders do not match the visual"));
            }
            last = end;
            for x in [start, v.token as u32, v.frames, v.height, v.width] {
                hash.update(x.to_le_bytes());
            }
            hash.update(&v.rgb);
            let mut out = vec![0.; v.tokens() as usize * 3072];
            let frame_bytes = (v.width * v.height * 3) as usize;
            let frame_values = out.len() / v.frames as usize;
            for (frame, rows) in v
                .rgb
                .chunks_exact(frame_bytes)
                .zip(out.chunks_exact_mut(frame_values))
            {
                let patches = v.patches(frame);
                let mut err = [0u8; 512];
                let rc = unsafe {
                    ds4_sys::ds4_bridge_hy4_encode_vision(
                        self.raw.as_ptr(),
                        patches.as_ptr(),
                        v.height / 16,
                        v.width / 16,
                        rows.as_mut_ptr(),
                        err.as_mut_ptr().cast(),
                        err.len(),
                    )
                };
                if rc != 0 {
                    return Err(crate::fail(rc, &err));
                }
            }
            spans.push(ds4_sys::ds4_hy4_span {
                start,
                frames: v.frames,
                height: v.height / 32,
                width: v.width / 32,
                token: v.token,
                features: out.as_ptr(),
            });
            features.push(out);
        }
        let tag = u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap()).max(1);
        let mut err = [0u8; 512];
        let rc = unsafe {
            ds4_sys::ds4_bridge_hy4_sync_media(
                self.raw.as_ptr(),
                ids.as_ptr(),
                ids.len() as i32,
                spans.as_ptr(),
                spans.len() as u32,
                tag,
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        self.host.invalidate();
        if rc != 0 {
            return Err(crate::fail(rc, &err));
        }
        self.host.replace_checkpoint(ids);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn processor_geometry_and_patch_order() {
        assert_eq!(image_size(33, 65).unwrap(), (64, 96));
        assert_eq!(video_size(320, 240, 4).unwrap(), (320, 224));
        assert!(video_size(1, 201, 1).is_err());
        let mut v = Visual {
            width: 64,
            height: 32,
            frames: 1,
            rgb: vec![0; 64 * 32 * 3],
            token: IMAGE,
        };
        v.rgb[(16 * 3) + 1] = 255;
        assert_eq!(v.tokens(), 3);
        let p = v.patches(&v.rgb);
        assert_eq!(p[768 + 256], 1.);
        assert_eq!(p[768], -1.);
    }
}
