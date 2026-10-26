//! The output stage: canvas rules, RGB24, the high-quality resize and the mux.
//!
//! Everything here is host-side and device-free, in the order the reference
//! runs it (`h3.c:1608-1700`): validate the requested geometry, decode to
//! F32 RGB, convert to RGB24 (`h3.c:731-742`), resize only when the output
//! canvas differs from the decoded one (`h3_host.c:532-635`), then mux frames
//! and soundtrack (`h3_ffmpeg.c:615-760`).
//!
//! The canvas contract is `h3_valid_params` (`h3.c:488-520`) plus the frame
//! minimum that generation adds on top of it (`h3.c:869-874`):
//!
//! - canvas: at least 32 and a multiple of 32, area at most `768*1344`
//!   (`h3_host.h:7-8`);
//! - render canvas: both dimensions or neither, at least 32, a multiple of
//!   32, no larger than the output canvas, and the same aspect ratio to the
//!   pixel (`rw * h == rh * w`);
//! - frames: requested at least 5, aligned to the 5 + 17k cadence, and a real
//!   render needs one 22-frame chunk.
//!
//! The latent canvas is the render canvas divided by the VAE spatial ratio 16
//! (`h3_host.c:43-46`), not the output canvas, which is why a smaller render
//! canvas is the way to spend less compute on the same picture geometry.

use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::minimax_h3::host::{align_frame_count, latent_canvas};

/// `h3_host.h:7`: every canvas edge is a multiple of it.
pub const CANVAS_MULTIPLE: i32 = 32;
/// `h3_host.h:8`: the released canvas pixel limit, 768 * 1344.
pub const MAX_PIXELS: i64 = 768 * 1344;
/// `h3_host.h:9`: the released frame rate, and the only one the mux writes.
pub const FPS: i32 = 24;
/// `h3.c:869-874`: a generation needs one trained 22-frame decoder chunk.
pub const MIN_GENERATION_FRAMES: i32 = 22;
/// `h3.c:515-519`: the aligned frame count must stay inside the release.
pub const MAX_ALIGNED_FRAMES: i32 = 362;

/// The requested geometry, validated the way `h3_valid_params` does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Canvas {
    width: i32,
    height: i32,
    render_width: i32,
    render_height: i32,
}

/// How strictly the delivered frame must follow the model canvas.
///
/// The reference serves the first two and refuses anything else
/// (`h3.c:505-520`). `Anamorphic` is the named delivery deviation: the frame
/// is an exact presentation ratio, the canvas is the model canvas, and the
/// resize stretches each axis by a different factor (`presets.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputRule {
    /// `h3_valid_params`: released area, multiples of 32, exact aspect.
    Released,
    /// A delivery frame of the same aspect, above the released area.
    Delivery,
    /// A delivery frame of any aspect: the scale is not uniform.
    Anamorphic,
}

impl Canvas {
    /// Validate a request. `render_width`/`render_height` of `0` mean "decode
    /// at the output canvas" (`h3.c:500-514`, `h3.c:863-866`).
    pub fn new(
        width: i32,
        height: i32,
        render_width: i32,
        render_height: i32,
    ) -> Result<Canvas, String> {
        Canvas::build(
            width,
            height,
            render_width,
            render_height,
            OutputRule::Released,
        )
    }

    /// The same validation with the delivery exception: the output may be
    /// larger than 768*1344 px, the render canvas still may not.
    pub fn new_delivery(
        width: i32,
        height: i32,
        render_width: i32,
        render_height: i32,
    ) -> Result<Canvas, String> {
        Canvas::build(
            width,
            height,
            render_width,
            render_height,
            OutputRule::Delivery,
        )
    }

    /// A delivery frame of any aspect. The reference refuses this
    /// (`h3.c:505-520`); the port names it as the anamorphic deviation and
    /// keeps every other canvas rule.
    pub fn new_anamorphic(
        width: i32,
        height: i32,
        render_width: i32,
        render_height: i32,
    ) -> Result<Canvas, String> {
        Canvas::build(
            width,
            height,
            render_width,
            render_height,
            OutputRule::Anamorphic,
        )
    }

    fn build(
        width: i32,
        height: i32,
        render_width: i32,
        render_height: i32,
        rule: OutputRule,
    ) -> Result<Canvas, String> {
        if width < CANVAS_MULTIPLE || height < CANVAS_MULTIPLE {
            return Err("width and height must be multiples of 32 and at least 32".to_string());
        }

        // The multiple-of-32 rule is the release's canvas rule. A delivery
        // frame such as 1080x1920 is not one, and the render canvas still is.
        if rule == OutputRule::Released
            && (width % CANVAS_MULTIPLE != 0 || height % CANVAS_MULTIPLE != 0)
        {
            return Err("width and height must be multiples of 32 and at least 32".to_string());
        }

        if rule == OutputRule::Released && i64::from(width) * i64::from(height) > MAX_PIXELS {
            return Err("canvas exceeds the released 768*1344 pixel limit".to_string());
        }

        if (render_width == 0) != (render_height == 0) {
            return Err("render width and height must be set together".to_string());
        }

        let (render_width, render_height) = if render_width == 0 {
            // The output is also the render canvas here, so the render rules
            // apply to it even in delivery mode.
            if rule != OutputRule::Released
                && (width % CANVAS_MULTIPLE != 0
                    || height % CANVAS_MULTIPLE != 0
                    || i64::from(width) * i64::from(height) > MAX_PIXELS)
            {
                return Err(
                    "a delivery canvas with no render size must be a released canvas".to_string(),
                );
            }

            (width, height)
        } else {
            let geometry_broken = render_width < CANVAS_MULTIPLE
                || render_height < CANVAS_MULTIPLE
                || render_width % CANVAS_MULTIPLE != 0
                || render_height % CANVAS_MULTIPLE != 0
                || render_width > width
                || render_height > height;
            // The cross-multiplication is the reference's aspect rule; an
            // anamorphic delivery is the one case that does not apply it.
            let aspect_broken = rule != OutputRule::Anamorphic
                && i64::from(render_width) * i64::from(height)
                    != i64::from(render_height) * i64::from(width);

            if geometry_broken || aspect_broken {
                // The reference reports one message for both failures. An
                // anamorphic canvas drops the aspect clause because that rule
                // does not apply to it.
                return Err(if rule == OutputRule::Anamorphic {
                    "internal render canvas must be multiples of 32 no larger than the output \
                     canvas"
                        .to_string()
                } else {
                    "internal render canvas must be same-aspect multiples of 32 \
                            no larger than the output canvas"
                        .to_string()
                });
            }

            // The render canvas is the model canvas, so the released area
            // governs it in every rule. A delivery frame above the cap does
            // not license a bigger canvas.
            if i64::from(render_width) * i64::from(render_height) > MAX_PIXELS {
                return Err(
                    "the render canvas exceeds the released 768*1344 pixel limit".to_string(),
                );
            }

            (render_width, render_height)
        };

        Ok(Canvas {
            width,
            height,
            render_width,
            render_height,
        })
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    pub fn render_width(&self) -> i32 {
        self.render_width
    }

    pub fn render_height(&self) -> i32 {
        self.render_height
    }

    /// The delivery scale per axis, `(x, y)`. Equal factors unless the frame
    /// aspect differs from the canvas.
    pub fn scale(&self) -> (f64, f64) {
        (
            f64::from(self.width) / f64::from(self.render_width),
            f64::from(self.height) / f64::from(self.render_height),
        )
    }

    /// Horizontal scale over vertical scale: 1.0 means square pixels, and any
    /// other value is the exact stretch the anamorphic delivery applies.
    pub fn distortion(&self) -> f64 {
        let (x, y) = self.scale();
        x / y
    }

    /// Whether the delivered frame has a different aspect from the canvas.
    pub fn is_anamorphic(&self) -> bool {
        i64::from(self.render_width) * i64::from(self.height)
            != i64::from(self.render_height) * i64::from(self.width)
    }

    /// The latent canvas the DiT denoises, from the render canvas
    /// (`h3.c:880`, `h3_host.c:43-46`).
    pub fn latent(&self) -> (i32, i32) {
        latent_canvas(self.render_width, self.render_height)
    }

    /// The spatial rope scale the reference applies (`h3.c:1489-1490`): 0.5
    /// only for a 256x256 render canvas, unless the reference rope is forced.
    /// Note it reads the render canvas, so a 256x256 render into a larger
    /// output canvas still takes 0.5.
    pub fn rope_scale(&self, use_reference_rope: bool) -> f32 {
        if !use_reference_rope && self.render_width == 256 && self.render_height == 256 {
            0.5
        } else {
            1.0
        }
    }
}

/// Validate a requested frame count and return the aligned one
/// (`h3.c:515-519`, `h3.c:869-874`, `h3_host.c:17-23`).
pub fn check_frames(requested: i32) -> Result<i32, String> {
    if requested < 5 {
        return Err("frames must align within the released 5..362 range".to_string());
    }

    let aligned = align_frame_count(requested);

    if aligned > MAX_ALIGNED_FRAMES {
        return Err("frames must align within the released 5..362 range".to_string());
    }

    if aligned < MIN_GENERATION_FRAMES {
        return Err("generation requires at least one trained 22-frame decoder chunk".to_string());
    }

    Ok(aligned)
}

/// Decoded F32 frames turned into the interleaved RGB24 the mux writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbFrames {
    frames: i32,
    width: i32,
    height: i32,
    rgb: Vec<u8>,
}

impl RgbFrames {
    /// `h3.c:1638-1645`: convert the decoder's frame-major F32 RGB in [0,1].
    pub fn from_f32(
        rgb: &[f32],
        frames: i32,
        width: i32,
        height: i32,
    ) -> Result<RgbFrames, String> {
        if frames < 1 || width < 1 || height < 1 {
            return Err("decoded frame geometry is empty".to_string());
        }

        let expected = frame_bytes(frames, width, height)?;

        if rgb.len() != expected {
            return Err(format!(
                "decoded frames hold {} values, expected {expected}",
                rgb.len()
            ));
        }

        Ok(RgbFrames {
            frames,
            width,
            height,
            rgb: rgb_f32_to_u8(rgb),
        })
    }

    pub fn frames(&self) -> i32 {
        self.frames
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    pub fn bytes(&self) -> &[u8] {
        &self.rgb
    }

    /// `h3.c:1647-1660`: resize only when the output canvas differs, and
    /// report the canvas the file will carry.
    pub fn resized_to(&self, width: i32, height: i32) -> Result<RgbFrames, String> {
        if width == self.width && height == self.height {
            return Ok(self.clone());
        }

        let rgb = resize_rgb24(
            &self.rgb,
            self.frames,
            self.width,
            self.height,
            width,
            height,
        )?;

        Ok(RgbFrames {
            frames: self.frames,
            width,
            height,
            rgb,
        })
    }
}

fn frame_bytes(frames: i32, width: i32, height: i32) -> Result<usize, String> {
    let pixels = usize::try_from(i64::from(width) * i64::from(height))
        .map_err(|_| "frame geometry overflows".to_string())?;
    let bytes = pixels
        .checked_mul(3)
        .and_then(|value| value.checked_mul(usize::try_from(frames).ok()?))
        .ok_or_else(|| "frame geometry overflows".to_string())?;
    Ok(bytes)
}

/// `h3.c:731-742`: scale by 255 in F32, clamp, round half to even, truncate.
pub fn rgb_f32_to_u8(rgb: &[f32]) -> Vec<u8> {
    rgb.iter()
        .map(|value| {
            let scaled = (value * 255.0).clamp(0.0, 255.0);
            scaled.round_ties_even() as u8
        })
        .collect()
}

/// `h3_host.c:532-635`: the non-Apple branch of the reference's
/// `h3_resize_rgb24_high_quality`, transcribed with its double-precision
/// bilinear kernel, half-pixel centers, edge clamping and round-half-to-even
/// write. The identity case is the reference's `memcpy`.
pub fn resize_rgb24(
    input: &[u8],
    frames: i32,
    input_width: i32,
    input_height: i32,
    output_width: i32,
    output_height: i32,
) -> Result<Vec<u8>, String> {
    if frames < 1 || input_width < 1 || input_height < 1 || output_width < 1 || output_height < 1 {
        return Err("cannot resize an empty frame set".to_string());
    }

    let input_bytes = frame_bytes(frames, input_width, input_height)?;

    if input.len() != input_bytes {
        return Err(format!(
            "resize input holds {} values, expected {input_bytes}",
            input.len()
        ));
    }

    let output_bytes = frame_bytes(frames, output_width, output_height)?;

    if input_width == output_width && input_height == output_height {
        return Ok(input.to_vec());
    }

    let mut output = vec![0u8; output_bytes];
    let input_frame =
        input_bytes / usize::try_from(frames).map_err(|_| "frame count".to_string())?;
    let output_frame =
        output_bytes / usize::try_from(frames).map_err(|_| "frame count".to_string())?;

    for frame in 0..frames as usize {
        let source = &input[frame * input_frame..(frame + 1) * input_frame];
        let destination = &mut output[frame * output_frame..(frame + 1) * output_frame];
        resize_one(
            source,
            destination,
            input_width,
            input_height,
            output_width,
            output_height,
        );
    }

    Ok(output)
}

/// One frame of the reference's bilinear scaler (`h3_host.c:607-632`).
fn resize_one(
    source: &[u8],
    destination: &mut [u8],
    input_width: i32,
    input_height: i32,
    output_width: i32,
    output_height: i32,
) {
    let input_width = input_width as usize;
    let output_width = output_width as usize;

    for y in 0..output_height as usize {
        let source_y = (y as f64 + 0.5) * input_height as f64 / output_height as f64 - 0.5;
        let mut y0 = source_y.floor() as i64;
        let mut fy = source_y - y0 as f64;

        if y0 < 0 {
            y0 = 0;
            fy = 0.0;
        }

        let mut y1 = y0 + 1;

        if y1 >= input_height as i64 {
            y1 = input_height as i64 - 1;
            fy = 0.0;
        }

        for x in 0..output_width {
            let source_x = (x as f64 + 0.5) * input_width as f64 / output_width as f64 - 0.5;
            let mut x0 = source_x.floor() as i64;
            let mut fx = source_x - x0 as f64;

            if x0 < 0 {
                x0 = 0;
                fx = 0.0;
            }

            let mut x1 = x0 + 1;

            if x1 >= input_width as i64 {
                x1 = input_width as i64 - 1;
                fx = 0.0;
            }

            for channel in 0..3 {
                let top_left =
                    f64::from(source[(y0 as usize * input_width + x0 as usize) * 3 + channel]);
                let top_right =
                    f64::from(source[(y0 as usize * input_width + x1 as usize) * 3 + channel]);
                let bottom_left =
                    f64::from(source[(y1 as usize * input_width + x0 as usize) * 3 + channel]);
                let bottom_right =
                    f64::from(source[(y1 as usize * input_width + x1 as usize) * 3 + channel]);

                let top = top_left * (1.0 - fx) + top_right * fx;
                let bottom = bottom_left * (1.0 - fx) + bottom_right * fx;
                let value = top * (1.0 - fy) + bottom * fy;
                destination[(y * output_width + x) * 3 + channel] =
                    value.clamp(0.0, 255.0).round_ties_even() as u8;
            }
        }
    }
}

/// `h3_ffmpeg.c:615-760`: interleaved F32 PCM, clamped to +-0.5 the way the
/// reference clamps it before `alimiter` sees it.
pub fn interleave_pcm(pcm: &[f32], samples: i32, channels: i32) -> Result<Vec<f32>, String> {
    if samples < 1 || channels < 1 {
        return Err("cannot interleave an empty waveform".to_string());
    }

    let channels_usize = usize::try_from(channels).map_err(|_| "channel count".to_string())?;

    if pcm.len() != samples as usize * channels_usize {
        return Err(format!(
            "waveform holds {} values, expected {}",
            pcm.len(),
            samples as usize * channels_usize
        ));
    }

    let mut interleaved = vec![0f32; pcm.len()];

    for sample in 0..samples as usize {
        for channel in 0..channels_usize {
            let value = pcm[channel * samples as usize + sample];
            interleaved[sample * channels_usize + channel] = value.clamp(-0.5, 0.5);
        }
    }

    Ok(interleaved)
}

/// The ffmpeg argument vector the reference builds (`h3_ffmpeg.c:679-694`),
/// kept separate so it can be compared with the reference line by line.
pub fn mux_args(
    path: &Path,
    width: i32,
    height: i32,
    sample_rate: i32,
    channels: i32,
    audio_fd: i32,
) -> Vec<String> {
    vec![
        "-y".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pixel_format".to_string(),
        "rgb24".to_string(),
        "-video_size".to_string(),
        format!("{width}x{height}"),
        "-framerate".to_string(),
        FPS.to_string(),
        "-i".to_string(),
        "pipe:0".to_string(),
        "-f".to_string(),
        "f32le".to_string(),
        "-ar".to_string(),
        sample_rate.to_string(),
        "-ac".to_string(),
        channels.to_string(),
        "-i".to_string(),
        format!("pipe:{audio_fd}"),
        "-map".to_string(),
        "0:v:0".to_string(),
        "-map".to_string(),
        "1:a:0".to_string(),
        "-c:v".to_string(),
        "libx264".to_string(),
        "-preset".to_string(),
        "fast".to_string(),
        "-crf".to_string(),
        "18".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-af".to_string(),
        "alimiter=limit=0.5:level=disabled".to_string(),
        "-c:a".to_string(),
        "aac".to_string(),
        "-b:a".to_string(),
        "192k".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
        path.to_string_lossy().into_owned(),
    ]
}

/// Mux RGB24 frames and a channel-major F32 waveform into one MP4 through
/// `ffmpeg`, exactly as `h3_ffmpeg.c:615-760` does it: the frames on stdin,
/// the PCM on a second inherited descriptor, both written from their own
/// threads so neither pipe can starve the other.
///
/// `program` is operator-configured; a generation request cannot choose it.
pub fn mux_mp4(
    program: &Path,
    path: &Path,
    frames: &RgbFrames,
    pcm: &[f32],
    samples: i32,
    channels: i32,
    sample_rate: i32,
) -> Result<(), String> {
    if frames.frames() < 1 || frames.width() < 2 || frames.height() < 2 {
        return Err("invalid FFmpeg A/V output arguments".to_string());
    }

    if frames.width() % 2 != 0 || frames.height() % 2 != 0 {
        return Err("invalid FFmpeg A/V output arguments".to_string());
    }

    if samples < 1 || channels < 1 || sample_rate < 1 {
        return Err("invalid FFmpeg A/V output arguments".to_string());
    }

    let interleaved = interleave_pcm(pcm, samples, channels)?;

    if let Some(parent) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }

    let audio_fd = MUX_AUDIO_FD;
    let args = mux_args(
        path,
        frames.width(),
        frames.height(),
        sample_rate,
        channels,
        audio_fd,
    );
    let (audio_read, audio_write) =
        io::pipe().map_err(|error| format!("cannot create FFmpeg A/V pipes: {error}"))?;
    let (video_read, mut video_write) =
        io::pipe().map_err(|error| format!("cannot create FFmpeg A/V pipes: {error}"))?;

    let mut command = Command::new(program);
    command
        .args(&args)
        .stdin(Stdio::from(video_read))
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    // The second input is a pipe, not a file: the child inherits the read end
    // on a descriptor past stdio, the way `h3_ffmpeg.c:703-708` places it.
    let read_fd = std::os::fd::AsRawFd::as_raw_fd(&audio_read);
    unsafe {
        use std::os::unix::process::CommandExt;
        command.pre_exec(move || {
            if libc::dup2(read_fd, audio_fd) < 0 {
                return Err(io::Error::last_os_error());
            }
            // dup2(fd, fd) leaves CLOEXEC set on io::pipe's original fd.
            if libc::fcntl(audio_fd, libc::F_SETFD, 0) < 0 {
                return Err(io::Error::last_os_error());
            }

            Ok(())
        });
    }

    let mut child = command
        .spawn()
        .map_err(|error| format!("cannot start FFmpeg: {error}"))?;

    // Release the parent's video reader so an exited child causes EPIPE.
    drop(command);
    drop(audio_read);

    // Drain stderr while the pipes are being written: a full stderr pipe would
    // block the child and turn a failed encode into a hang.
    let mut stderr = child.stderr.take();
    let errors = std::thread::spawn(move || {
        use std::io::Read;

        let mut text = String::new();

        if let Some(stream) = stderr.as_mut() {
            let _ = stream.read_to_string(&mut text);
        }

        text
    });

    let rgb = frames.bytes();
    let (video_result, audio_result) = std::thread::scope(|scope| {
        // Both write ends are moved into their threads and dropped when the
        // thread returns; a borrowed handle would keep the pipe open and
        // ffmpeg would wait for an EOF that never comes.
        let video = scope.spawn(move || {
            video_write
                .write_all(rgb)
                .map_err(|error| format!("cannot write FFmpeg video pipe: {error}"))
        });
        let audio = scope.spawn(move || {
            let mut sink = io::BufWriter::new(audio_write);

            for value in interleaved.iter() {
                sink.write_all(&value.to_le_bytes())
                    .map_err(|error| format!("cannot write FFmpeg audio pipe: {error}"))?;
            }

            sink.flush()
                .map_err(|error| format!("cannot write FFmpeg audio pipe: {error}"))
        });

        let video = video
            .join()
            .unwrap_or_else(|_| Err("video writer panicked".into()));
        let audio = audio
            .join()
            .unwrap_or_else(|_| Err("audio writer panicked".into()));

        (video, audio)
    });

    let status = child
        .wait()
        .map_err(|error| format!("cannot wait for FFmpeg: {error}"))?;
    let text = errors.join().unwrap_or_default();
    let detail = if text.trim().is_empty() {
        String::new()
    } else {
        format!(": {}", text.trim())
    };

    video_result?;
    audio_result?;

    if !status.success() {
        let code = status
            .code()
            .map(|code| code.to_string())
            .unwrap_or_else(|| "signal".to_string());
        return Err(format!("FFmpeg exited with status {code}{detail}"));
    }

    Ok(())
}

/// The descriptor the child is given for the audio pipe, above stdio
/// (`h3_ffmpeg.c:659-665`).
const MUX_AUDIO_FD: i32 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mux_pipe_lifetimes() {
        const CHILD: &str = "DS4_H3_MUX_CHILD";
        if let Ok(mode) = std::env::var(CHILD) {
            assert_eq!(unsafe { libc::fcntl(MUX_AUDIO_FD, libc::F_GETFD) }, -1);
            let edge = if mode == "early-exit" { 256 } else { 32 };
            let frames = RgbFrames::from_f32(
                &vec![0.25; 3 * 2 * edge * edge],
                2,
                edge as i32,
                edge as i32,
            )
            .unwrap();
            let path = std::env::temp_dir().join(format!("h3-mux-{}.mp4", std::process::id()));
            let program = if mode == "early-exit" {
                "/bin/false"
            } else {
                "ffmpeg"
            };
            let result = mux_mp4(
                Path::new(program),
                &path,
                &frames,
                &vec![0.0; 2 * 3200],
                3200,
                2,
                32000,
            );
            let _ = std::fs::remove_file(path);
            if mode == "early-exit" {
                assert!(result.is_err());
            } else {
                assert!(result.is_ok(), "fd 3 must survive exec: {result:?}");
            }
            return;
        }

        use std::os::unix::process::CommandExt;
        for mode in ["early-exit", "audio-fd"] {
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "minimax_h3::render::tests::mux_pipe_lifetimes",
                    "--nocapture",
                ])
                .env(CHILD, mode)
                .stdout(Stdio::null())
                .process_group(0)
                .spawn()
                .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "{mode}: {status}");
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    unsafe {
                        libc::kill(-(child.id() as i32), libc::SIGKILL);
                    }
                    child.wait().unwrap();
                    panic!("{mode}: mux blocked after child exit");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }

    /// The phone reel: the model renders 576x1024, the file is 1080x1920.
    /// The reference refuses that output (`h3.c:497-500`); the delivery
    /// constructor is the named exception, and everything else still holds.
    #[test]
    fn delivery_canvas_allows_an_upscaled_output() {
        let canvas = Canvas::new_delivery(1080, 1920, 576, 1024).expect("the phone reel");
        assert_eq!(canvas.render_width(), 576);
        assert_eq!(canvas.render_height(), 1024);
        assert_eq!(canvas.latent(), latent_canvas(576, 1024));

        assert!(Canvas::new(1080, 1920, 576, 1024).is_err());
        // Wrong aspect, a render canvas larger than the output, and a
        // non-canonical render canvas all stay refused.
        assert!(Canvas::new_delivery(1080, 1920, 640, 1024).is_err());
        assert!(Canvas::new_delivery(1080, 1920, 1152, 2048).is_err());
        assert!(Canvas::new_delivery(1080, 1920, 600, 1067).is_err());
    }

    #[test]
    fn anamorphic_canvas_stretches_each_axis_by_a_named_factor() {
        // The platform's own 768P 16:9 file is 1344x768 (1.75), so a 16:9
        // presentation frame from that canvas is a 1.6 % non-uniform scale.
        let canvas = Canvas::new_anamorphic(1920, 1080, 1344, 768).expect("16:9 from 768P");
        let (x, y) = canvas.scale();
        assert!((x - 1.428_571_428_571_428_6).abs() < 1e-12);
        assert!((y - 1.406_25).abs() < 1e-12);
        assert!((canvas.distortion() - 1.015_873_015_873_015_9).abs() < 1e-12);
        assert!(canvas.is_anamorphic());

        // The same delivery under the reference's rules is refused.
        assert!(Canvas::new_delivery(1920, 1080, 1344, 768).is_err());
        assert!(Canvas::new(1920, 1080, 1344, 768).is_err());

        // An exact-aspect delivery is not anamorphic.
        let reel = Canvas::new_anamorphic(576, 1024, 288, 512).expect("9:16");
        assert!(!reel.is_anamorphic());
        assert!((reel.distortion() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn anamorphic_canvas_keeps_every_other_canvas_rule() {
        // The render canvas may not be larger than the output.
        assert_eq!(
            Canvas::new_anamorphic(1080, 1920, 1152, 2048).unwrap_err(),
            "internal render canvas must be multiples of 32 no larger than the output canvas"
        );
        // Nor leave the released grid.
        assert!(Canvas::new_anamorphic(1080, 1920, 600, 1067).is_err());
        // A delivery frame above the released area does not license a bigger
        // model canvas.
        assert_eq!(
            Canvas::new_anamorphic(1080, 1920, 768, 1408).unwrap_err(),
            "the render canvas exceeds the released 768*1344 pixel limit"
        );
        // The output itself still needs at least 32 pixels per edge.
        assert!(Canvas::new_anamorphic(16, 1920, 0, 0).is_err());
    }

    #[test]
    fn canvas_rejects_edges_outside_the_release() {
        assert!(Canvas::new(32, 32, 0, 0).is_ok());
        assert_eq!(
            Canvas::new(31, 64, 0, 0).unwrap_err(),
            "width and height must be multiples of 32 and at least 32"
        );
        assert_eq!(
            Canvas::new(48, 64, 0, 0).unwrap_err(),
            "width and height must be multiples of 32 and at least 32"
        );
        assert!(Canvas::new(1344, 768, 0, 0).is_ok());
        assert_eq!(
            Canvas::new(1408, 768, 0, 0).unwrap_err(),
            "canvas exceeds the released 768*1344 pixel limit"
        );
        assert_eq!(
            Canvas::new(1344, 800, 0, 0).unwrap_err(),
            "canvas exceeds the released 768*1344 pixel limit"
        );
    }

    #[test]
    fn render_canvas_keeps_the_aspect_to_the_pixel() {
        assert!(Canvas::new(1024, 768, 512, 384).is_ok());
        assert!(Canvas::new(256, 256, 0, 0).is_ok());
        assert_eq!(
            Canvas::new(1024, 768, 512, 0).unwrap_err(),
            "render width and height must be set together"
        );
        assert_eq!(
            Canvas::new(1024, 768, 512, 512).unwrap_err(),
            "internal render canvas must be same-aspect multiples of 32 no larger \
             than the output canvas"
        );
        assert_eq!(
            Canvas::new(1024, 768, 2048, 1536).unwrap_err(),
            "internal render canvas must be same-aspect multiples of 32 no larger \
             than the output canvas"
        );
        assert_eq!(
            Canvas::new(1024, 768, 508, 384).unwrap_err(),
            "internal render canvas must be same-aspect multiples of 32 no larger \
             than the output canvas"
        );
    }

    #[test]
    fn latent_canvas_comes_from_the_render_canvas() {
        let canvas = Canvas::new(1344, 768, 672, 384).unwrap();
        assert_eq!(canvas.latent(), (42, 24));
    }

    #[test]
    fn rope_scale_is_half_only_at_256() {
        assert_eq!(Canvas::new(256, 256, 0, 0).unwrap().rope_scale(false), 0.5);
        assert_eq!(Canvas::new(256, 256, 0, 0).unwrap().rope_scale(true), 1.0);
        assert_eq!(
            Canvas::new(512, 512, 256, 256).unwrap().rope_scale(false),
            0.5
        );
        assert_eq!(Canvas::new(512, 384, 0, 0).unwrap().rope_scale(false), 1.0);
    }

    #[test]
    fn frame_alignment_matches_the_reference_window() {
        assert_eq!(check_frames(22).unwrap(), 22);
        assert_eq!(check_frames(23).unwrap(), 39);
        assert_eq!(check_frames(362).unwrap(), 362);
        assert_eq!(
            check_frames(5).unwrap_err(),
            "generation requires at least one trained 22-frame decoder chunk"
        );
        assert_eq!(
            check_frames(4).unwrap_err(),
            "frames must align within the released 5..362 range"
        );
        assert_eq!(
            check_frames(363).unwrap_err(),
            "frames must align within the released 5..362 range"
        );
    }

    #[test]
    fn rgb_conversion_rounds_half_to_even() {
        let values = [0.0, 1.0, 0.5, -1.0, 2.0, 1.0 / 255.0];
        assert_eq!(rgb_f32_to_u8(&values), vec![0, 255, 128, 0, 255, 1]);
    }

    #[test]
    fn resize_identity_copies_and_upscale_interpolates() {
        let frame = [0u8, 255, 0, 255, 0, 0, 0, 0, 255, 255, 255, 255];
        assert_eq!(resize_rgb24(&frame, 1, 2, 2, 2, 2).unwrap(), frame);

        let constant = [7u8; 12];
        let upscaled = resize_rgb24(&constant, 1, 2, 2, 4, 4).unwrap();
        assert!(upscaled.iter().all(|value| *value == 7));

        let doubled = resize_rgb24(&frame, 1, 2, 2, 4, 4).unwrap();
        assert_eq!(doubled.len(), 48);
        // The corners are the source corners, the middle a real blend.
        assert_eq!(&doubled[0..3], &[0, 255, 0]);
        assert_eq!(&doubled[45..48], &[255, 255, 255]);
    }

    #[test]
    fn interleave_writes_sample_major_and_clamps() {
        let pcm = [1.0, -1.0, 0.25, -0.25];
        let interleaved = interleave_pcm(&pcm, 2, 2).unwrap();
        assert_eq!(interleaved, vec![0.5, 0.25, -0.5, -0.25]);
    }

    #[test]
    fn mux_args_match_the_reference_spelling() {
        let args = mux_args(Path::new("out.mp4"), 512, 288, 32000, 2, 3);
        assert_eq!(args[0], "-y");
        assert!(args.windows(2).any(|pair| pair == ["-crf", "18"]));
        assert!(args.windows(2).any(|pair| pair == ["-i", "pipe:3"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["-af", "alimiter=limit=0.5:level=disabled"]));
        assert_eq!(args[args.len() - 1], "out.mp4");
    }
}
