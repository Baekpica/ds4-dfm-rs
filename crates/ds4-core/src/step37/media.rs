//! Step's official image crop layout, before pixel conversion or GPU work.
use crate::{Error, Result};

const MAX_EDGE: u32 = 32768;
const MAX_IMAGE: u32 = 3024;
const BASE_EDGE: u32 = 728;
const PATCH_EDGE: u32 = 504;
const BASE_TOKENS: u32 = 169;
const PATCH_TOKENS: u32 = 81;

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

#[cfg(test)]
mod tests {
    use super::*;
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
