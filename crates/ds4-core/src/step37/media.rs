//! Step's official image crop layout and CPU pixel conversion.
use super::resample::{resize_float, resize_rgb};
use crate::{Error, Result};
use image::RgbImage;

const MAX_EDGE: u32 = 32768;
const MAX_IMAGE: u32 = 3024;
const BASE_EDGE: u32 = 728;
const PATCH_EDGE: u32 = 504;
const BASE_TOKENS: u32 = 169;
const PATCH_TOKENS: u32 = 81;
const MAX_RGB_BYTES: u64 = 128 * 1024 * 1024;
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
            let image = image::open(directory.join(format!("{name}.png")))
                .unwrap()
                .into_rgb8();
            let (plan, crops) = image_crops(&image, MAX_MEDIA_TOKENS).unwrap();
            assert_eq!(crops.len(), v["crops"].as_array().unwrap().len());
            assert_eq!(crops.len(), plan.boxes.len() + 1);
            let mut worst = 0.0f32;
            for (index, (crop, expected)) in
                crops.iter().zip(v["crops"].as_array().unwrap()).enumerate()
            {
                let stem = expected["stem"].as_str().unwrap();
                let rgb = std::fs::read(directory.join(format!("{stem}.rgb"))).unwrap();
                assert_eq!(crop.as_raw(), &rgb, "{stem} RGB bytes");
                let pixels =
                    normalized_pixels(crop, if index == 0 { BASE_EDGE } else { PATCH_EDGE });
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
