//! The processor has two contracts: Pillow RGB8 resize, then Torch F32 AA.
//! Pins and complete reference generation are in the Step fixture directory.
use image::RgbImage;

const CHANNELS: usize = 3;
const PIL_PRECISION: u32 = 22;

struct Weights<T> {
    start: usize,
    values: Vec<T>,
}

fn rgb_weights(input: usize, output: usize) -> Vec<Weights<u32>> {
    let scale = input as f64 / output as f64;
    let support = scale.max(1.0);
    let inverse = 1.0 / support;
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let values: Vec<f64> = (start..end)
                .map(|j| (1.0 - ((j as f64 - center + 0.5) * inverse).abs()).max(0.0))
                .collect();
            let total: f64 = values.iter().sum();
            Weights {
                start,
                values: values
                    .into_iter()
                    .map(|v| (v / total * f64::from(1u32 << PIL_PRECISION)).round() as u32)
                    .collect(),
            }
        })
        .collect()
}

fn float_weights(input: usize, output: usize) -> Vec<Weights<f32>> {
    let scale = input as f32 / output as f32;
    let support = scale.max(1.0);
    let inverse = 1.0 / support;
    (0..output)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale;
            let start = ((center - support + 0.5) as i64).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let mut values: Vec<f32> = (start..end)
                .map(|j| (1.0 - ((j as f32 - center + 0.5) * inverse).abs()).max(0.0))
                .collect();
            let total: f32 = values.iter().sum();
            for v in &mut values {
                *v /= total;
            }
            Weights { start, values }
        })
        .collect()
}

pub(super) fn resize_rgb(input: &RgbImage, width: u32, height: u32) -> RgbImage {
    let iw = input.width() as usize;
    let ih = input.height() as usize;
    let ow = width as usize;
    let oh = height as usize;
    assert!(iw > 0 && ih > 0 && ow > 0 && oh > 0);
    // Pillow rounds the horizontal pass back to eight bits before vertical
    // filtering. A single floating resize gives a different crop image.
    let horizontal = if iw == ow {
        input.as_raw().clone()
    } else {
        let weights = rgb_weights(iw, ow);
        let mut out = vec![0u8; CHANNELS * ow * ih];
        for y in 0..ih {
            for (x, w) in weights.iter().enumerate() {
                for c in 0..CHANNELS {
                    let mut sum = 1u64 << (PIL_PRECISION - 1);
                    for (i, &weight) in w.values.iter().enumerate() {
                        sum += u64::from(input.as_raw()[(y * iw + w.start + i) * CHANNELS + c])
                            * u64::from(weight);
                    }
                    out[(y * ow + x) * CHANNELS + c] = (sum >> PIL_PRECISION).min(255) as u8;
                }
            }
        }
        out
    };
    if ih == oh {
        return RgbImage::from_raw(width, height, horizontal).unwrap();
    }
    let weights = rgb_weights(ih, oh);
    let mut out = vec![0u8; CHANNELS * ow * oh];
    for (y, w) in weights.iter().enumerate() {
        for x in 0..ow {
            for c in 0..CHANNELS {
                let mut sum = 1u64 << (PIL_PRECISION - 1);
                for (i, &weight) in w.values.iter().enumerate() {
                    sum += u64::from(horizontal[((w.start + i) * ow + x) * CHANNELS + c])
                        * u64::from(weight);
                }
                out[(y * ow + x) * CHANNELS + c] = (sum >> PIL_PRECISION).min(255) as u8;
            }
        }
    }
    RgbImage::from_raw(width, height, out).unwrap()
}

pub(super) fn resize_float(input: &[f32], shape: [usize; 2], output: [usize; 2]) -> Vec<f32> {
    let [iw, ih] = shape;
    let [ow, oh] = output;
    assert!(iw > 0 && ih > 0 && ow > 0 && oh > 0 && input.len() == CHANNELS * iw * ih);
    // Torch AA uses float coefficients and separable horizontal/vertical
    // filtering. Preserve CHW and normalize before calling this operation.
    let horizontal = if iw == ow {
        input.to_vec()
    } else {
        let weights = float_weights(iw, ow);
        let mut out = vec![0.0; CHANNELS * ow * ih];
        for c in 0..CHANNELS {
            for y in 0..ih {
                for (x, w) in weights.iter().enumerate() {
                    let row = c * iw * ih + y * iw + w.start;
                    out[c * ow * ih + y * ow + x] = w
                        .values
                        .iter()
                        .enumerate()
                        .map(|(i, weight)| input[row + i] * weight)
                        .sum();
                }
            }
        }
        out
    };
    if ih == oh {
        return horizontal;
    }
    let weights = float_weights(ih, oh);
    let mut out = vec![0.0; CHANNELS * ow * oh];
    for c in 0..CHANNELS {
        for (y, w) in weights.iter().enumerate() {
            for x in 0..ow {
                let start = c * ow * ih + w.start * ow + x;
                out[c * ow * oh + y * ow + x] = w
                    .values
                    .iter()
                    .enumerate()
                    .map(|(i, weight)| horizontal[start + i * ow] * weight)
                    .sum();
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_resize_vectors() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/step37/resize-vectors.json"
        ))
        .unwrap();
        let input: Vec<u8> = (0..7 * 5 * 3).map(|v| (v * 2) as u8).collect();
        let image = RgbImage::from_raw(7, 5, input.clone()).unwrap();
        // The normalized-tensor resize consumes CHW floats, not RGB8 or HWC.
        let tensor: Vec<f32> = (0..3)
            .flat_map(|c| (0..35).map(move |i| (i * 3 + c) as f32 * 2.0 / 255.0))
            .collect();
        for v in reference["vectors"].as_array().unwrap() {
            let width = v["output"][0].as_u64().unwrap() as u32;
            let height = v["output"][1].as_u64().unwrap() as u32;
            let expected: Vec<u8> = v["rgb"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            assert_eq!(resize_rgb(&image, width, height).into_raw(), expected);
            let actual = resize_float(&tensor, [7, 5], [width as usize, height as usize]);
            for (i, (got, want)) in actual
                .iter()
                .zip(v["float"].as_array().unwrap())
                .enumerate()
            {
                assert!(
                    (f64::from(*got) - want.as_f64().unwrap()).abs() < 1e-6,
                    "{width}x{height} element {i}: {got} vs {want}"
                );
            }
        }
    }
}
