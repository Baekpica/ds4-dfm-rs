//! Graphical-terminal output for the denoise preview (`h3_terminal.c`).
//!
//! The reference detects a Kitty/Ghostty or iTerm2-family terminal
//! (`h3_terminal.c:29-40`), encodes the raw RGB frame, and writes it with the
//! terminal's own escape sequence (`:121-164`, `:255-286`). The preview is
//! display only: it never touches the frames that are muxed into the file,
//! and the render gate is a hash identical with the preview on and off.
//!
//! One deviation, named: the iTerm2 path encodes PNG in-process with the
//! `image` crate instead of spawning FFmpeg through a temporary raw file
//! (`h3_terminal.c:177-228`). The bytes handed to the terminal are an
//! equivalent PNG of the same pixels; the display is not part of any gate.

use std::io::Write;
use std::sync::atomic::{AtomicU16, Ordering};

/// The reference's default zoom (`h3_terminal.c:19`).
const DEFAULT_ZOOM: u16 = 2;
/// Kitty payload chunks (`h3_terminal.c:137`).
const KITTY_CHUNK: usize = 4096;

static ZOOM: AtomicU16 = AtomicU16::new(DEFAULT_ZOOM);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    None,
    Kitty,
    Iterm2,
}

/// `h3_terminal_detect` (`:29-40`).
pub fn detect() -> Protocol {
    let set = |name: &str| std::env::var_os(name).is_some_and(|value| !value.is_empty());

    if set("KITTY_WINDOW_ID") || set("GHOSTTY_RESOURCES_DIR") {
        return Protocol::Kitty;
    }
    let program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    if program == "iTerm.app"
        || set("ITERM_SESSION_ID")
        || set("WEZTERM_PANE")
        || set("KONSOLE_VERSION")
    {
        return Protocol::Iterm2;
    }
    Protocol::None
}

/// `h3_terminal_protocol_name` (`:42-50`).
pub fn protocol_name(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Kitty => "Kitty graphics",
        Protocol::Iterm2 => "iTerm2 inline images",
        Protocol::None => "none",
    }
}

/// `h3_terminal_set_zoom` (`:48-52`): zero is refused, as the CLI does.
pub fn set_zoom(zoom: u16) -> bool {
    if zoom < 1 {
        return false;
    }
    ZOOM.store(zoom, Ordering::Relaxed);
    true
}

/// `h3_terminal_display_dimensions` (`:54-62`): the logical size the terminal
/// is asked for, which the reference scales by the zoom.
pub fn display_dimensions(width: u32, height: u32) -> Option<(u32, u32)> {
    let zoom = u32::from(ZOOM.load(Ordering::Relaxed));
    Some((width.checked_mul(zoom)?, height.checked_mul(zoom)?))
}

/// `base64_encode` (`:64-91`): the standard alphabet, padded.
fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(4 * data.len().div_ceil(3));

    for group in data.chunks(3) {
        let a = u32::from(group[0]);
        let b = u32::from(*group.get(1).unwrap_or(&0));
        let c = u32::from(*group.get(2).unwrap_or(&0));
        let triple = (a << 16) | (b << 8) | c;
        output.push(TABLE[(triple >> 18) as usize & 63] as char);
        output.push(TABLE[(triple >> 12) as usize & 63] as char);
        output.push(if group.len() > 1 {
            TABLE[(triple >> 6) as usize & 63] as char
        } else {
            '='
        });
        output.push(if group.len() > 2 {
            TABLE[triple as usize & 63] as char
        } else {
            '='
        });
    }

    output
}

/// `kitty_display` (`:121-164`): `f=24` raw RGB, `a=T` transmit and display,
/// the logical width and height in cells-free pixels, one `m` bit per chunk.
fn kitty_display(rgb: &[u8], width: u32, height: u32) -> Result<(), String> {
    let (display_width, display_height) =
        display_dimensions(width, height).ok_or("invalid Kitty display dimensions")?;
    let encoded = base64_encode(rgb);
    // Both encodings are ASCII, so byte slicing is character slicing.
    let bytes = encoded.as_bytes();
    let mut stdout = std::io::stdout().lock();
    let mut offset = 0usize;
    let mut first = true;

    while offset < bytes.len() {
        let chunk = (bytes.len() - offset).min(KITTY_CHUNK);
        let more = usize::from(offset + chunk < bytes.len());
        let header = if first {
            first = false;
            format!(
                "\x1b_Ga=T,f=24,t=d,s={width},v={height},w={display_width},h={display_height},m={more};"
            )
        } else {
            format!("\x1b_Gm={more};")
        };
        stdout
            .write_all(header.as_bytes())
            .and_then(|_| stdout.write_all(&bytes[offset..offset + chunk]))
            .and_then(|_| stdout.write_all(b"\x1b\\"))
            .map_err(|error| format!("cannot write Kitty frame: {error}"))?;
        offset += chunk;
    }

    stdout
        .write_all(b"\n")
        .and_then(|_| stdout.flush())
        .map_err(|error| format!("cannot write Kitty frame: {error}"))
}

/// `iterm2_display` (`:255-286`): an inline PNG as a base64 payload.
fn iterm2_display(rgb: &[u8], width: u32, height: u32) -> Result<(), String> {
    let (display_width, display_height) =
        display_dimensions(width, height).ok_or("invalid iTerm2 display dimensions")?;
    let image = image::RgbImage::from_raw(width, height, rgb.to_vec())
        .ok_or("iTerm2 frame does not match its dimensions")?;
    let mut png = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut png, image::ImageFormat::Png)
        .map_err(|error| format!("cannot encode iTerm2 PNG: {error}"))?;
    let encoded = base64_encode(png.get_ref());

    let mut stdout = std::io::stdout().lock();
    write!(
        stdout,
        "\x1b]1337;File=inline=1;width={display_width}px;height={display_height}px:"
    )
    .and_then(|_| stdout.write_all(encoded.as_bytes()))
    .and_then(|_| stdout.write_all(b"\x07\n"))
    .and_then(|_| stdout.flush())
    .map_err(|error| format!("cannot write iTerm2 frame: {error}"))
}

/// `h3_terminal_display_rgb24` (`:288-304`): packed RGB24, protocol dispatch.
pub fn display_rgb24(
    protocol: Protocol,
    rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<(), String> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("invalid terminal RGB frame")?;
    if rgb.len() != expected {
        return Err("invalid terminal RGB frame".to_string());
    }

    match protocol {
        Protocol::Kitty => kitty_display(rgb, width, height),
        Protocol::Iterm2 => iterm2_display(rgb, width, height),
        Protocol::None => Err("no supported graphical terminal detected".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 4648 test vectors: the reference's encoder pads exactly this way
    /// (`h3_terminal.c:86-89`).
    #[test]
    fn base64_pads_like_the_reference() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    /// `h3_terminal.c:19`: Retina terminals get twice the logical size.
    #[test]
    fn zoom_scales_the_display_size() {
        assert!(set_zoom(2));
        assert_eq!(display_dimensions(512, 192), Some((1024, 384)));
        assert!(!set_zoom(0));
        assert_eq!(display_dimensions(512, 192), Some((1024, 384)));
    }
}
