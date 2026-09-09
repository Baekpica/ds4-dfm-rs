//! Source Inkling media preprocessing before the native encoders.

use std::io::Cursor;

use image::{metadata::Orientation, DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};

use crate::{Error, Result, VisionImageInfo};

const PATCH: u32 = 40;
const TIMES: u32 = 2;
const CHANNELS: u32 = 3;
const PIXEL_SCALE: f32 = 255.0;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
const MAX_ENCODED: usize = 32 * 1024 * 1024;
const MAX_DECODED: u64 = 128 * 1024 * 1024;
const MAX_EDGE: u32 = 32768;
const MAX_PATCHES: u32 = 8192;

pub(crate) struct ImagePixels {
    pub(crate) pixels: Vec<f32>,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
}

fn image_error(message: &str) -> Error {
    Error {
        code: 1,
        message: message.into(),
    }
}

fn decode_error(error: image::ImageError) -> Error {
    image_error(&format!("Inkling image decode failed: {error}"))
}

fn image_decoder(data: &[u8]) -> Result<impl ImageDecoder + '_> {
    if data.is_empty() || data.len() > MAX_ENCODED {
        return Err(image_error("Inkling image is empty or exceeds 32 MiB"));
    }
    let format = image::guess_format(data).map_err(decode_error)?;
    if !matches!(format, ImageFormat::Png | ImageFormat::Jpeg) {
        return Err(image_error("Inkling image requires PNG or JPEG"));
    }
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_EDGE);
    limits.max_image_height = Some(MAX_EDGE);
    limits.max_alloc = Some(MAX_DECODED);
    let mut reader = ImageReader::with_format(Cursor::new(data), format);
    reader.limits(limits);
    let decoder = reader.into_decoder().map_err(decode_error)?;
    // Some decoder allocation limits are advisory; bound the pixel buffer too.
    if decoder.total_bytes() > MAX_DECODED {
        return Err(image_error("Inkling decoded image exceeds 128 MiB"));
    }
    Ok(decoder)
}

fn image_geometry(width: u32, height: u32, budget: u32) -> Result<VisionImageInfo> {
    if width == 0 || height == 0 || width > MAX_EDGE || height > MAX_EDGE {
        return Err(image_error("invalid Inkling image dimensions"));
    }
    // The source adds an empty right column when width is a multiple of 40.
    let rows = height.div_ceil(PATCH);
    let cols = width / PATCH + 1;
    let count = u64::from(rows) * u64::from(cols);
    if count > u64::from(budget.min(MAX_PATCHES)) {
        return Err(image_error(
            "Inkling image exceeds the media token budget (maximum 8192)",
        ));
    }
    Ok(VisionImageInfo {
        source_width: width,
        source_height: height,
        content_width: width,
        content_height: height,
        padded_width: cols * PATCH,
        padded_height: rows * PATCH,
        grid_width: cols,
        grid_height: rows,
        token_count: count as u32,
    })
}

fn oriented_geometry(
    decoder: &impl ImageDecoder,
    orientation: Orientation,
    budget: u32,
) -> Result<VisionImageInfo> {
    let (mut width, mut height) = decoder.dimensions();
    if matches!(
        orientation,
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    ) {
        std::mem::swap(&mut width, &mut height);
    }
    image_geometry(width, height, budget)
}

pub(crate) fn probe_image(data: &[u8]) -> Result<VisionImageInfo> {
    let mut decoder = image_decoder(data)?;
    let orientation = decoder.orientation().map_err(decode_error)?;
    oriented_geometry(&decoder, orientation, MAX_PATCHES)
}

pub(crate) fn prepare_image(data: &[u8], budget: u32) -> Result<ImagePixels> {
    let mut decoder = image_decoder(data)?;
    let orientation = decoder.orientation().map_err(decode_error)?;
    // Match the source PIL loading path: EXIF transform, then RGB conversion.
    oriented_geometry(&decoder, orientation, budget)?;
    let mut image = DynamicImage::from_decoder(decoder).map_err(decode_error)?;
    image.apply_orientation(orientation);
    let rgb = image.into_rgb8();
    prepare_rgb(rgb.as_raw(), rgb.width(), rgb.height(), budget)
}

pub(crate) fn prepare_rgb(rgb: &[u8], width: u32, height: u32, budget: u32) -> Result<ImagePixels> {
    let input_bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|v| v.checked_mul(u64::from(CHANNELS)));
    if width == 0 || height == 0 || input_bytes != Some(rgb.len() as u64) {
        return Err(image_error("invalid Inkling RGB dimensions or byte count"));
    }
    let info = image_geometry(width, height, budget)?;
    let (rows, cols) = (info.grid_height, info.grid_width);
    let count = u64::from(info.token_count);
    let values = usize::try_from(count * u64::from(TIMES * PATCH * PATCH * CHANNELS))
        .map_err(|_| image_error("Inkling image workspace is too large"))?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(values)
        .map_err(|_| image_error("cannot allocate Inkling image workspace"))?;
    // TorchvisionBackend fuses rescaling into the FP32 mean/std tensors.
    // Keep this order, including raw -1 padding, before the encoder BF16 cast.
    let mean = MEAN.map(|v| v * PIXEL_SCALE);
    let std = STD.map(|v| v * PIXEL_SCALE);
    for py in 0..rows {
        for px in 0..cols {
            let start = pixels.len();
            for site in 0..PATCH * PATCH {
                let (ix, iy) = (px * PATCH + site % PATCH, py * PATCH + site / PATCH);
                for c in 0..CHANNELS as usize {
                    let value = if ix < width && iy < height {
                        rgb[((u64::from(iy) * u64::from(width) + u64::from(ix))
                            * u64::from(CHANNELS)) as usize
                            + c] as f32
                    } else {
                        -1.0
                    };
                    pixels.push((value - mean[c]) / std[c]);
                }
            }
            // Static images repeat the same patch along the two-frame axis.
            pixels.extend_from_within(start..);
        }
    }
    Ok(ImagePixels { pixels, rows, cols })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> Vec<[u32; 3]> {
        let json: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/inkling/image-normalize.json"
        ))
        .unwrap();
        json["bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| std::array::from_fn(|c| row[c].as_u64().unwrap() as u32))
            .collect()
    }

    #[test]
    fn image_source_geometry_and_bits() {
        let reference = reference();
        // Include both exact-multiple axes and partial right/bottom patches.
        for (width, height, cols, rows) in [
            (1, 1, 1, 1),
            (40, 40, 2, 1),
            (41, 39, 2, 1),
            (39, 41, 1, 2),
            (80, 81, 3, 3),
            (256, 2, 7, 1),
        ] {
            let rgb: Vec<_> = (0..width * height * 3)
                .map(|i| ((i * 37 + 17) % 256) as u8)
                .collect();
            let got = prepare_rgb(&rgb, width, height, rows * cols).unwrap();
            assert_eq!((got.rows, got.cols), (rows, cols));
            assert_eq!(got.pixels.len(), (rows * cols * 2 * 40 * 40 * 3) as usize);
            for patch in 0..rows * cols {
                for time in 0..2 {
                    for y in 0..40 {
                        for x in 0..40 {
                            let (ix, iy) = (patch % cols * 40 + x, patch / cols * 40 + y);
                            for c in 0..3 {
                                let value = if ix < width && iy < height {
                                    usize::from(rgb[((iy * width + ix) * 3 + c) as usize]) + 1
                                } else {
                                    0
                                };
                                let at =
                                    ((((patch * 2 + time) * 40 + y) * 40 + x) * 3 + c) as usize;
                                assert_eq!(
                                    got.pixels[at].to_bits(),
                                    reference[value][c as usize],
                                    "{width}x{height}, patch{patch}, t{time}, y{y}, x{x}, c{c}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn image_rejects_invalid_size_and_budget() {
        assert!(prepare_rgb(&[], 0, 1, 1).is_err());
        assert!(prepare_rgb(&[], 1, 0, 1).is_err());
        assert!(prepare_rgb(&[], u32::MAX, u32::MAX, u32::MAX).is_err());
        assert!(prepare_rgb(&[0; 2], 1, 1, 1).is_err());
        assert!(prepare_rgb(&[0; 4], 1, 1, 1).is_err());
        assert!(prepare_rgb(&[0; 3], 1, 1, 0).is_err());
        // An exact-width image includes the extra empty column in admission.
        assert!(prepare_rgb(&vec![0; 40 * 40 * 3], 40, 40, 1).is_err());
    }

    fn png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
        use image::ImageEncoder;
        let mut out = Vec::new();
        image::codecs::png::PngEncoder::new(&mut out)
            .write_image(rgba, width, height, image::ExtendedColorType::Rgba8)
            .unwrap();
        out
    }

    #[test]
    fn decode_png_rgb_and_alpha() {
        let data = png(40, 1, &[12, 64, 231, 0].repeat(40));
        let info = probe_image(&data).unwrap();
        assert_eq!(
            (info.source_width, info.source_height, info.token_count),
            (40, 1, 2)
        );
        let got = prepare_image(&data, 2).unwrap();
        let want = prepare_rgb(&[12, 64, 231].repeat(40), 40, 1, 2).unwrap();
        assert_eq!(got.pixels, want.pixels);
        assert!(prepare_image(&data, 1).is_err());
        assert!(prepare_image(&data[..40], 2).is_err());
        assert!(probe_image(b"not an image").is_err());
        let too_wide = png(32769, 1, &[0, 0, 0, 255].repeat(32769));
        assert!(probe_image(&too_wide).is_err());
    }

    #[test]
    fn decode_jpeg_exif_orientation() {
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 100)
            .encode(&[64; 40 * 3], 40, 1, image::ExtendedColorType::Rgb8)
            .unwrap();
        let plain = probe_image(&jpeg).unwrap();
        assert_eq!(
            (plain.source_width, plain.source_height, plain.token_count),
            (40, 1, 2)
        );
        // Little-endian TIFF IFD: orientation 6 means rotate 90 degrees CW.
        let exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        let mut rotated = jpeg[..2].to_vec();
        rotated.extend_from_slice(&[0xff, 0xe1]);
        rotated.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
        rotated.extend_from_slice(exif);
        rotated.extend_from_slice(&jpeg[2..]);
        let info = probe_image(&rotated).unwrap();
        assert_eq!(
            (info.source_width, info.source_height, info.token_count),
            (1, 40, 1)
        );
        let got = prepare_image(&rotated, 1).unwrap();
        assert_eq!((got.rows, got.cols), (1, 1));
        assert_eq!(
            got.pixels,
            prepare_rgb(&[64; 40 * 3], 1, 40, 1).unwrap().pixels
        );
    }
}
