//! Step's official image crop layout and CPU pixel conversion.
use super::resample::{resize_float, resize_rgb};
use crate::{Error, Result};
use image::{
    metadata::Orientation, DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits, RgbImage,
};
use std::io::Cursor;

const MAX_EDGE: u32 = 32768;
const MAX_IMAGE: u32 = 3024;
const BASE_EDGE: u32 = 728;
const PATCH_EDGE: u32 = 504;
const BASE_TOKENS: u32 = 169;
const PATCH_TOKENS: u32 = 81;
const MAX_RGB_BYTES: u64 = 128 * 1024 * 1024;
const MAX_ENCODED: usize = 32 * 1024 * 1024;
const IMAGE_START: i32 = 128000;
const IMAGE_TOKEN: i32 = 128001;
const IMAGE_END: i32 = 128002;
const PATCH_START: i32 = 128003;
const PATCH_NEWLINE: i32 = 128004;
const PATCH_END: i32 = 128005;
const MAX_MEDIA_TOKENS: u32 = 8192;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

#[derive(Debug)]
struct ImagePlan {
    padded: [u32; 2],
    base: [u32; 2],
    crop: [u32; 2],
    window: u32,
    grid: [u32; 2],
    boxes: Vec<[u32; 4]>,
    tokens: u32,
}

fn image_plan(width: u32, height: u32) -> Result<ImagePlan> {
    if width == 0 || height == 0 || width > MAX_EDGE || height > MAX_EDGE {
        return Err(Error {
            code: 1,
            message: "invalid Step image dimensions".into(),
        });
    }
    let long = width.max(height);
    let short = width.min(height);
    let padded = if short < 32 && long > 4 * short {
        [long, long]
    } else {
        [width, height]
    };
    let scale = f64::from(MAX_IMAGE) / f64::from(padded[0].max(padded[1]));
    let base = if scale < 1.0 {
        padded.map(|edge| (f64::from(edge) * scale) as u32)
    } else {
        padded
    };
    let long = base[0].max(base[1]);
    let short = base[0].min(base[1]);
    let window = if long <= BASE_EDGE {
        if f64::from(long) / f64::from(short) > 1.5 {
            short
        } else {
            0
        }
    } else if long > 4 * short {
        short.min(PATCH_EDGE)
    } else {
        PATCH_EDGE
    };
    let mut plan = ImagePlan {
        padded,
        base,
        crop: base,
        window,
        grid: [0, 0],
        boxes: Vec::new(),
        tokens: BASE_TOKENS + 2,
    };
    if window == 0 {
        return Ok(plan);
    }
    // Preserve the source's floating threshold and truncation at 0.2.
    plan.crop = base.map(|edge| {
        if edge < window {
            return edge;
        }
        let ratio = f64::from(edge) / f64::from(window);
        let whole = edge / window;
        window * (whole + u32::from(ratio - f64::from(whole) > 0.2))
    });
    plan.grid = plan.crop.map(|edge| edge.div_ceil(window));
    for y in 0..plan.grid[1] {
        for x in 0..plan.grid[0] {
            let left = (x * window).min(plan.crop[0].saturating_sub(window));
            let top = (y * window).min(plan.crop[1].saturating_sub(window));
            plan.boxes.push([left, top, window, window]);
        }
    }
    plan.tokens += plan.boxes.len() as u32 * (PATCH_TOKENS + 2) + plan.grid[1] - 1;
    Ok(plan)
}

fn check_plan(plan: &ImagePlan, budget: u32) -> Result<()> {
    if plan.tokens > budget.min(MAX_MEDIA_TOKENS) {
        return Err(Error {
            code: 1,
            message: "Step image exceeds the media token budget (maximum 8192)".into(),
        });
    }
    for [width, height] in [plan.padded, plan.base, plan.crop] {
        if u64::from(width) * u64::from(height) * 3 > MAX_RGB_BYTES {
            return Err(Error {
                code: 1,
                message: "Step RGB buffer exceeds 128 MiB".into(),
            });
        }
    }
    Ok(())
}

fn image_crops(input: &RgbImage, budget: u32) -> Result<(ImagePlan, Vec<RgbImage>)> {
    let plan = image_plan(input.width(), input.height())?;
    check_plan(&plan, budget)?;
    let base = if input.dimensions() == (plan.padded[0], plan.padded[1]) {
        resize_rgb(input, plan.base[0], plan.base[1])
    } else {
        let mut padded = RgbImage::new(plan.padded[0], plan.padded[1]);
        image::imageops::replace(&mut padded, input, 0, 0);
        resize_rgb(&padded, plan.base[0], plan.base[1])
    };
    let mut crops = Vec::with_capacity(1 + plan.boxes.len());
    if plan.boxes.is_empty() {
        crops.push(base);
        return Ok((plan, crops));
    }
    let crop_source = resize_rgb(&base, plan.crop[0], plan.crop[1]);
    crops.push(base);
    for &[x, y, width, height] in &plan.boxes {
        // PIL.crop fills out-of-bounds rows with black. A short dimension
        // may be smaller than the 504-wide crop even when the grid has a row.
        let mut crop = RgbImage::new(width, height);
        let copied_width = width.min(crop_source.width().saturating_sub(x));
        let copied_height = height.min(crop_source.height().saturating_sub(y));
        for row in 0..copied_height {
            let source = ((y + row) * crop_source.width() + x) as usize * 3;
            let destination = (row * width) as usize * 3;
            let bytes = copied_width as usize * 3;
            crop.as_mut()[destination..destination + bytes]
                .copy_from_slice(&crop_source.as_raw()[source..source + bytes]);
        }
        crops.push(crop);
    }
    Ok((plan, crops))
}

// Core owns these host buffers until synchronous native encoding completes.
pub(crate) struct Step37Crop {
    pub(crate) pixels: Vec<f32>,
    pub(crate) offset: u32,
    pub(crate) rows: u32,
    pub(crate) edge: u32,
}

pub(crate) struct Step37Media {
    pub(crate) crops: Vec<Step37Crop>,
}

impl Step37Media {
    pub(crate) fn probe(data: &[u8]) -> Result<crate::VisionImageInfo> {
        let (decoder, orientation, plan) = image_source(data, MAX_MEDIA_TOKENS)?;
        let (width, height) = oriented_size(decoder.dimensions(), orientation);
        Ok(crate::VisionImageInfo {
            source_width: width,
            source_height: height,
            content_width: plan.base[0],
            content_height: plan.base[1],
            padded_width: plan.padded[0],
            padded_height: plan.padded[1],
            grid_width: plan.grid[0],
            grid_height: plan.grid[1],
            token_count: plan.tokens,
        })
    }

    pub(crate) fn tokens(data: &[u8]) -> Result<Vec<i32>> {
        let (_, _, plan) = image_source(data, MAX_MEDIA_TOKENS)?;
        Ok(image_layout(&plan).tokens)
    }

    pub(crate) fn prepare(tokens: &[i32], images: &[crate::VisionInput<'_>]) -> Result<Self> {
        if images.is_empty() || images.len() > 4 {
            return Err(image_error("Step vision requires 1 to 4 images"));
        }
        let mut layouts = Vec::with_capacity(images.len());
        let mut previous_end = 0;
        let mut budget = MAX_MEDIA_TOKENS;
        // Validate every payload header, complete replacement and the aggregate
        // budget before any image decode or normalized-pixel allocation.
        for image in images {
            let (_, _, plan) = image_source(image.data, budget)?;
            budget -= plan.tokens;
            let layout = image_layout(&plan);
            let start = image.token_offset as usize;
            let end = start
                .checked_add(layout.tokens.len())
                .filter(|&end| {
                    start >= previous_end && end <= tokens.len() && u32::try_from(end).is_ok()
                })
                .ok_or_else(|| image_error("invalid Step image token span"))?;
            if tokens[start..end] != layout.tokens
                || tokens[previous_end..start].contains(&IMAGE_TOKEN)
            {
                return Err(image_error("Step image tokens do not match the payload"));
            }
            previous_end = end;
            layouts.push(layout);
        }
        if tokens[previous_end..].contains(&IMAGE_TOKEN) {
            return Err(image_error("Step image token has no payload"));
        }
        let mut prepared = Self { crops: Vec::new() };
        for (image, layout) in images.iter().zip(layouts) {
            let (_, rgb) = decode_image(image.data, layout.tokens.len() as u32)?;
            let (_, mut crops) = image_crops(&rgb, layout.tokens.len() as u32)?;
            drop(rgb);
            for span in layout.crops {
                let crop = std::mem::take(&mut crops[span.index]);
                prepared.crops.push(Step37Crop {
                    pixels: normalized_pixels(&crop, span.edge),
                    offset: image.token_offset + span.offset as u32,
                    rows: span.rows,
                    edge: span.edge,
                });
            }
        }
        Ok(prepared)
    }
}

struct CropSpan {
    index: usize,
    offset: usize,
    rows: u32,
    edge: u32,
}

struct ImageLayout {
    tokens: Vec<i32>,
    crops: Vec<CropSpan>,
}

fn image_layout(plan: &ImagePlan) -> ImageLayout {
    let mut layout = ImageLayout {
        tokens: Vec::with_capacity(plan.tokens as usize),
        crops: Vec::with_capacity(plan.boxes.len() + 1),
    };
    // Processor order is row-major patches, row separators, then the base
    // image. RGB crops keep base at index zero, so retain that explicit map.
    for index in 0..plan.boxes.len() {
        layout.tokens.push(PATCH_START);
        layout.crops.push(CropSpan {
            index: index + 1,
            offset: layout.tokens.len(),
            rows: PATCH_TOKENS,
            edge: PATCH_EDGE,
        });
        layout
            .tokens
            .extend(std::iter::repeat_n(IMAGE_TOKEN, PATCH_TOKENS as usize));
        layout.tokens.push(PATCH_END);
        if (index + 1) % plan.grid[0] as usize == 0 && index + 1 < plan.boxes.len() {
            layout.tokens.push(PATCH_NEWLINE);
        }
    }
    layout.tokens.push(IMAGE_START);
    layout.crops.push(CropSpan {
        index: 0,
        offset: layout.tokens.len(),
        rows: BASE_TOKENS,
        edge: BASE_EDGE,
    });
    layout
        .tokens
        .extend(std::iter::repeat_n(IMAGE_TOKEN, BASE_TOKENS as usize));
    layout.tokens.push(IMAGE_END);
    layout
}

fn image_error(message: impl Into<String>) -> Error {
    Error {
        code: 1,
        message: message.into(),
    }
}

fn decode_error(error: image::ImageError) -> Error {
    image_error(format!("Step image decode failed: {error}"))
}

fn image_source(
    data: &[u8],
    budget: u32,
) -> Result<(impl ImageDecoder + '_, Orientation, ImagePlan)> {
    if data.is_empty() || data.len() > MAX_ENCODED {
        return Err(image_error("Step image is empty or exceeds 32 MiB"));
    }
    let format = image::guess_format(data).map_err(decode_error)?;
    if !matches!(format, ImageFormat::Png | ImageFormat::Jpeg) {
        return Err(image_error("Step image requires PNG or JPEG"));
    }
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_EDGE);
    limits.max_image_height = Some(MAX_EDGE);
    limits.max_alloc = Some(MAX_RGB_BYTES);
    let mut reader = ImageReader::with_format(Cursor::new(data), format);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(decode_error)?;
    if decoder.total_bytes() > MAX_RGB_BYTES {
        return Err(image_error("Step decoded image exceeds 128 MiB"));
    }
    if decoder.color_type().bits_per_pixel() / u16::from(decoder.color_type().channel_count()) != 8
    {
        return Err(image_error("Step image requires 8-bit channels"));
    }
    let orientation = decoder.orientation().map_err(decode_error)?;
    let (width, height) = oriented_size(decoder.dimensions(), orientation);
    let plan = image_plan(width, height)?;
    check_plan(&plan, budget)?;
    Ok((decoder, orientation, plan))
}

fn oriented_size(size: (u32, u32), orientation: Orientation) -> (u32, u32) {
    let (mut width, mut height) = size;
    if matches!(
        orientation,
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    ) {
        std::mem::swap(&mut width, &mut height);
    }
    (width, height)
}

fn decode_image(data: &[u8], budget: u32) -> Result<(ImagePlan, RgbImage)> {
    let (decoder, orientation, plan) = image_source(data, budget)?;
    // Match the byte-loading path's EXIF transpose followed by RGB conversion.
    // Converting RGBA to RGB drops alpha; it does not composite a background.
    let mut image = DynamicImage::from_decoder(decoder).map_err(decode_error)?;
    image.apply_orientation(orientation);
    Ok((plan, image.into_rgb8()))
}

fn normalized_pixels(image: &RgbImage, edge: u32) -> Vec<f32> {
    let area = image.width() as usize * image.height() as usize;
    let mut values = vec![0.0; area * 3];
    for channel in 0..3 {
        for i in 0..area {
            values[channel * area + i] =
                (f32::from(image.as_raw()[i * 3 + channel]) / 255.0 - MEAN[channel]) / STD[channel];
        }
    }
    resize_float(
        &values,
        [image.width() as usize, image.height() as usize],
        [edge as usize; 2],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_media_checks_all_spans() {
        let data = png(33, 27, [12, 64, 231, 0]);
        let layout = Step37Media::tokens(&data).unwrap();
        let mut tokens = vec![10];
        tokens.extend_from_slice(&layout);
        tokens.push(10);
        let second = tokens.len() as u32;
        tokens.extend_from_slice(&layout);
        let images = [
            crate::VisionInput {
                data: &data,
                token_offset: 1,
            },
            crate::VisionInput {
                data: &data,
                token_offset: second,
            },
        ];
        let media = Step37Media::prepare(&tokens, &images).unwrap();
        assert_eq!(media.crops.len(), 2);
        assert_eq!(media.crops[0].offset, 2);
        assert_eq!(media.crops[1].offset, second + 1);
        assert_eq!(media.crops[0].pixels, media.crops[1].pixels);
        assert_eq!(media.crops[0].rows, BASE_TOKENS);
        assert_eq!(media.crops[0].edge, BASE_EDGE);
        assert_eq!(
            media.crops[0].pixels.len(),
            3 * BASE_EDGE as usize * BASE_EDGE as usize
        );
        assert!(Step37Media::prepare(&tokens, &images[..1]).is_err());
        assert!(Step37Media::prepare(&tokens, &[images[1], images[0]]).is_err());
        assert!(Step37Media::prepare(&tokens, &[images[0], images[0]]).is_err());
        assert!(Step37Media::prepare(&tokens[..tokens.len() - 1], &images).is_err());
        let mut corrupt = tokens.clone();
        corrupt[2] = 10;
        assert!(Step37Media::prepare(&corrupt, &images).is_err());
        tokens.push(IMAGE_TOKEN);
        assert!(Step37Media::prepare(&tokens, &images).is_err());
        assert!(Step37Media::prepare(&tokens, &[]).is_err());
    }

    #[test]
    fn media_budget_is_shared() {
        let data = png(3000, 32, [0; 4]);
        let layout = Step37Media::tokens(&data).unwrap();
        assert!(layout.len() > MAX_MEDIA_TOKENS as usize / 2);
        let tokens = [layout.as_slice(), layout.as_slice()].concat();
        let images = [
            crate::VisionInput {
                data: &data,
                token_offset: 0,
            },
            crate::VisionInput {
                data: &data,
                token_offset: layout.len() as u32,
            },
        ];
        assert!(Step37Media::prepare(&tokens, &images).is_err());
    }

    #[test]
    fn official_media_layout() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/step37/media-vectors.json"
        ))
        .unwrap();
        for vector in reference["vectors"].as_array().unwrap() {
            let width = vector["source"][0].as_u64().unwrap() as u32;
            let height = vector["source"][1].as_u64().unwrap() as u32;
            let plan = image_plan(width, height).unwrap();
            let layout = image_layout(&plan);
            let expected: Vec<i32> = vector["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            assert_eq!(layout.tokens, expected, "{width}x{height}");
            assert_eq!(layout.tokens.len(), plan.tokens as usize);
            assert_eq!(layout.crops.len(), plan.boxes.len() + 1);
            for crop in &layout.crops {
                assert!(layout.tokens[crop.offset..crop.offset + crop.rows as usize]
                    .iter()
                    .all(|&token| token == IMAGE_TOKEN));
            }
            assert_eq!(layout.crops.last().unwrap().index, 0);
            assert_eq!(layout.crops.last().unwrap().rows, BASE_TOKENS);
        }
    }

    fn png(width: u32, height: u32, pixel: [u8; 4]) -> Vec<u8> {
        use image::ImageEncoder;
        let mut data = Vec::new();
        image::codecs::png::PngEncoder::new(&mut data)
            .write_image(
                &pixel.repeat((width * height) as usize),
                width,
                height,
                image::ExtendedColorType::Rgba8,
            )
            .unwrap();
        data
    }

    #[test]
    fn decode_preflights_and_drops_alpha() {
        let image = png(33, 27, [12, 64, 231, 0]);
        let (plan, rgb) = decode_image(&image, 171).unwrap();
        assert_eq!(plan.tokens, 171);
        assert_eq!(rgb.as_raw(), &[12, 64, 231].repeat(33 * 27));
        assert!(decode_image(&image, 170).is_err());
        assert!(decode_image(&image[..32], 171).is_err());
        assert!(decode_image(b"not an image", 8192).is_err());
        assert!(decode_image(&png(32769, 1, [0; 4]), 8192).is_err());
        assert!(decode_image(&png(32768, 1, [0; 4]), 8192).is_err());
        assert!(decode_image(&png(32768, 32, [0; 4]), 8192).is_err());
    }

    #[test]
    fn decode_jpeg_applies_exif() {
        let pixels: Vec<u8> = (0..20)
            .flat_map(|y| (0..40).flat_map(move |x| [x * 6, y * 12, (x + y) * 4]))
            .collect();
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 100)
            .encode(&pixels, 40, 20, image::ExtendedColorType::Rgb8)
            .unwrap();
        let (_, original) = decode_image(&jpeg, 8192).unwrap();
        let exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        let mut rotated = jpeg[..2].to_vec();
        rotated.extend_from_slice(&[0xff, 0xe1]);
        rotated.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
        rotated.extend_from_slice(exif);
        rotated.extend_from_slice(&jpeg[2..]);
        let (plan, rgb) = decode_image(&rotated, 8192).unwrap();
        assert_eq!(rgb.dimensions(), (20, 40));
        assert_eq!(plan.base, [20, 40]);
        for y in 0..20 {
            for x in 0..40 {
                assert_eq!(rgb.get_pixel(19 - y, x), original.get_pixel(x, y));
            }
        }
    }

    #[test]
    fn budgets_precede_pixel_allocation() {
        assert!(check_plan(&image_plan(32768, 32).unwrap(), u32::MAX).is_err());
        assert!(check_plan(&image_plan(1, 32768).unwrap(), u32::MAX).is_err());
        assert!(check_plan(&image_plan(728, 728).unwrap(), 170).is_err());
        assert!(check_plan(&image_plan(728, 728).unwrap(), 171).is_ok());
    }

    #[test]
    #[ignore = "requires full pixels from make_pixel_vectors.py"]
    fn official_image_pixels() {
        let directory = std::path::PathBuf::from(
            std::env::var("STEP37_PIXEL_REF").expect("set STEP37_PIXEL_REF"),
        );
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("reference.json")).unwrap())
                .unwrap();
        for v in reference["vectors"].as_array().unwrap() {
            let name = v["name"].as_str().unwrap();
            let encoded = std::fs::read(directory.join(format!("{name}.png"))).unwrap();
            let (_, image) = decode_image(&encoded, MAX_MEDIA_TOKENS).unwrap();
            let (plan, crops) = image_crops(&image, MAX_MEDIA_TOKENS).unwrap();
            let tokens = Step37Media::tokens(&encoded).unwrap();
            let prepared = Step37Media::prepare(
                &tokens,
                &[crate::VisionInput {
                    data: &encoded,
                    token_offset: 0,
                }],
            )
            .unwrap();
            assert_eq!(crops.len(), v["crops"].as_array().unwrap().len());
            assert_eq!(crops.len(), plan.boxes.len() + 1);
            let mut worst = 0.0f32;
            for (index, (crop, expected)) in
                crops.iter().zip(v["crops"].as_array().unwrap()).enumerate()
            {
                let stem = expected["stem"].as_str().unwrap();
                let rgb = std::fs::read(directory.join(format!("{stem}.rgb"))).unwrap();
                assert_eq!(crop.as_raw(), &rgb, "{stem} RGB bytes");
                let slot = if index == 0 {
                    prepared.crops.len() - 1
                } else {
                    index - 1
                };
                let pixels = &prepared.crops[slot].pixels;
                let bytes = std::fs::read(directory.join(format!("{stem}.f32"))).unwrap();
                assert_eq!(bytes.len(), pixels.len() * 4);
                for (&got, raw) in pixels.iter().zip(bytes.chunks_exact(4)) {
                    let want = f32::from_le_bytes(raw.try_into().unwrap());
                    let error = (got - want).abs();
                    worst = worst.max(error);
                    assert!(
                        got.is_finite() && want.is_finite() && error <= 1e-6,
                        "{stem}: pixel difference {error}, {got} vs {want}"
                    );
                }
            }
            eprintln!(
                "{name}: {} crops RGB exact, full float max_abs={worst}",
                crops.len()
            );
        }
    }
    #[test]
    fn official_image_geometry() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/step37/image-plans.json"
        ))
        .unwrap();
        for v in reference["vectors"].as_array().unwrap() {
            let p = image_plan(
                v["source"][0].as_u64().unwrap() as u32,
                v["source"][1].as_u64().unwrap() as u32,
            )
            .unwrap();
            for (name, actual) in [
                ("padded", p.padded),
                ("base", p.base),
                ("crop", p.crop),
                ("grid", p.grid),
            ] {
                assert_eq!(
                    serde_json::json!(actual),
                    v[name],
                    "{} / {name}",
                    v["source"]
                );
            }
            assert_eq!(serde_json::json!(p.boxes), v["boxes"], "{}", v["source"]);
            assert_eq!(p.window, v["window"].as_u64().unwrap() as u32);
            assert_eq!(p.tokens, v["tokens"].as_u64().unwrap() as u32);
        }
        assert!(image_plan(0, 728).is_err());
        assert!(image_plan(1, MAX_EDGE + 1).is_err());
    }
}
