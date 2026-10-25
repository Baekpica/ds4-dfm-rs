//! MiniMax-M3.1's vision tower against the handoff's reference
//! (tools/minimax_vision_ref.py, rerun on the release's vision part): the
//! reference's pixel values in, the engine's language rows out. A BF16 tower
//! sits a BF16-noise distance from the truth, so both the engine and the BF16
//! reference are judged against the same reference run in F32: the engine
//! passes when its relative RMS to F32 is within NOISE_BAND of the BF16
//! reference's own and its mean cosine within COS_SLACK of it. Also checks
//! this crate's preprocessing of the image against the reference pixels.
//! Usage: minimax_vision MODEL REF_DIR IMAGE  (REF_DIR: shapes.json,
//! pixels.bin, embeddings.bin (BF16 run), embeddings_f32.bin (F32 run))
use ds4_core::{minimax_media, Backend, Model, ModelFamily};
use serde_json::Value;

// On test1.png (504 x 728): BF16 reference to F32 relative RMS 0.0302,
// mean cosine 0.99883; the engine 0.0316 and 0.99860.
const NOISE_BAND: f64 = 1.25;
const COS_SLACK: f64 = 0.001;

fn floats(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Mean and minimum row cosine and relative RMS of `a` against `b` ([n][6144]).
fn compare(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut sum, mut min, mut err2, mut ref2) = (0.0f64, f64::MAX, 0.0f64, 0.0f64);
    let n = b.len() / 6144;
    for r in 0..n {
        let (x, y) = (&a[r * 6144..(r + 1) * 6144], &b[r * 6144..(r + 1) * 6144]);
        let (mut xy, mut xx, mut yy) = (0.0f64, 0.0f64, 0.0f64);
        for (u, v) in x.iter().zip(y) {
            let (u, v) = (f64::from(*u), f64::from(*v));
            xy += u * v;
            xx += u * u;
            yy += v * v;
            err2 += (u - v) * (u - v);
            ref2 += v * v;
        }
        let cos = xy / (xx.sqrt() * yy.sqrt()).max(1e-30);
        sum += cos;
        min = min.min(cos);
    }
    (sum / n as f64, min, (err2 / ref2.max(1e-30)).sqrt())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: minimax_vision MODEL REF_DIR IMAGE".into());
    }
    let dir = std::path::Path::new(&args[1]);
    let shapes: Value = serde_json::from_slice(&std::fs::read(dir.join("shapes.json"))?)?;
    let grid: Vec<u32> = shapes["grid"]
        .as_array()
        .ok_or("grid")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    let pixels = floats(&dir.join("pixels.bin"))?;
    let want = floats(&dir.join("embeddings.bin"))?;
    let reference = minimax_media::Visual {
        patches: pixels.clone(),
        grid_t: grid[0],
        grid_h: grid[1],
        grid_w: grid[2],
    };
    // Preprocessing: PIL bicubic here, the fast processor's torchvision resize there.
    let ours = minimax_media::image(&std::fs::read(&args[2])?)?;
    let same_grid = (ours.grid_t, ours.grid_h, ours.grid_w) == (grid[0], grid[1], grid[2]);
    let (mut worst, mut differ) = (0.0f32, 0usize);
    if same_grid {
        for (a, b) in ours.patches.iter().zip(&pixels) {
            worst = worst.max((a - b).abs());
            differ += usize::from(a != b);
        }
    }
    println!(
        "preprocess: grid {:?} {}, {differ} of {} values differ, max |d| {worst:.4}",
        (ours.grid_t, ours.grid_h, ours.grid_w),
        if same_grid { "matches" } else { "DIFFERS" },
        pixels.len()
    );
    let model = Model::open(&args[0], Backend::Cuda, 8, true)?;
    if model.family() != ModelFamily::MinimaxVq {
        return Err("this check requires the MiniMax-M3.1 VQ release".into());
    }
    let session = model.session(4096)?;
    let started = std::time::Instant::now();
    let rows = session.minimax_vision_rows(&reference)?;
    let elapsed = started.elapsed().as_secs_f64();
    // MINIMAX_VISION_DUMP=FILE keeps the engine's rows (F32) for offline comparison.
    if let Ok(path) = std::env::var("MINIMAX_VISION_DUMP") {
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes)?;
    }
    if rows.len() != want.len() {
        return Err(format!("rows {} against reference {}", rows.len(), want.len()).into());
    }
    let f32 = floats(&dir.join("embeddings_f32.bin"))?;
    let (engine, floor) = (compare(&rows, &f32), compare(&want, &f32));
    let pass = engine.2 <= NOISE_BAND * floor.2 && engine.0 >= floor.0 - COS_SLACK;
    println!("tower: {} rows in {elapsed:.2} s; to F32: engine cosine mean {:.6} min {:.6} rel RMS {:.5}, \
        BF16 reference {:.6} / {:.6} / {:.5}: {}", rows.len() / 6144, engine.0, engine.1, engine.2, floor.0, floor.1,
        floor.2, if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
