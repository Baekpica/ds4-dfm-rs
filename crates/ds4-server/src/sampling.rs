//! Request-omitted sampling values and request seeds.
//!
//! Precedence, as vLLM (`--generation-config auto`) and SGLang
//! (`--sampling-defaults model`): request > server flags > the model's
//! `generation_config.json` (else GGUF `general.sampling.*`, llama.cpp's
//! keys) > OpenAI defaults. A request without a seed draws a fresh one
//! (llama.cpp seed -1, vLLM's global generator); `--seed` makes that
//! sequence reproducible.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Values a request may omit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
}

impl Default for SamplingDefaults {
    /// OpenAI's: temperature 1, nucleus off, no top-k or min-p.
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
        }
    }
}

/// One source's values; unset fields defer to the next source.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplingSource {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    /// `max_new_tokens`: vLLM's default `max_tokens`.
    pub max_tokens: Option<i32>,
}

impl SamplingDefaults {
    pub fn with(mut self, s: &SamplingSource) -> Self {
        self.temperature = s.temperature.unwrap_or(self.temperature);
        self.top_p = s.top_p.unwrap_or(self.top_p);
        self.top_k = s.top_k.unwrap_or(self.top_k);
        self.min_p = s.min_p.unwrap_or(self.min_p);
        self
    }
}

const GENERATION_CONFIG: &str = "generation_config.json";

/// The model's recommended values: `generation_config.json` beside the
/// model (a release directory, or a GGUF's directory), else the GGUF's
/// `general.sampling.*`. `do_sample` is ignored, as vLLM does.
pub fn model_source(model: &Path) -> SamplingSource {
    let dir = if model.is_dir() {
        Some(model)
    } else {
        model.parent()
    };
    if let Some(src) = dir.and_then(|d| json_source(&d.join(GENERATION_CONFIG))) {
        return src;
    }
    if !model.is_file() {
        return SamplingSource::default();
    }
    let Ok(g) = ds4_core::GgufFile::open(model) else {
        return SamplingSource::default();
    };
    SamplingSource {
        temperature: g.get_f32_compat("general.sampling.temp"),
        top_p: g.get_f32_compat("general.sampling.top_p"),
        top_k: g.get_u32("general.sampling.top_k").map(|k| k as i32),
        min_p: g.get_f32_compat("general.sampling.min_p"),
        max_tokens: None,
    }
}

fn json_source(path: &Path) -> Option<SamplingSource> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
    let i = |k: &str| v.get(k).and_then(|x| x.as_i64()).map(|x| x as i32);
    Some(SamplingSource {
        temperature: f("temperature"),
        top_p: f("top_p"),
        top_k: i("top_k").filter(|&k| k > 0),
        min_p: f("min_p"),
        max_tokens: i("max_new_tokens").filter(|&n| n > 0),
    })
}

/// splitmix64 state behind every server-drawn request seed.
static SEEDS: AtomicU64 = AtomicU64::new(0);
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

/// Seed the request-seed sequence: `--seed`, else process entropy.
pub fn init_seeds(seed: Option<u64>) {
    let base = seed.unwrap_or_else(|| {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        h.write_u32(std::process::id());
        h.finish()
    });
    SEEDS.store(base, Ordering::Relaxed);
}

/// A fresh seed for a request that names none.
pub fn next_seed() -> u64 {
    let mut z = SEEDS
        .fetch_add(GOLDEN, Ordering::Relaxed)
        .wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A request's `seed` field: non-negative values are the stream; a
/// negative one (llama.cpp's -1) asks for a fresh seed.
pub fn request_seed(v: f64) -> u64 {
    if v >= 0.0 {
        v as u64
    } else {
        next_seed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_apply_in_order() {
        let model = SamplingSource {
            temperature: Some(1.0),
            top_p: Some(0.95),
            ..Default::default()
        };
        let flags = SamplingSource {
            temperature: Some(0.6),
            ..Default::default()
        };
        let d = SamplingDefaults::default().with(&model).with(&flags);
        assert_eq!(
            (d.temperature, d.top_p, d.top_k, d.min_p),
            (0.6, 0.95, 0, 0.0)
        );
    }

    #[test]
    fn generation_config_supplies_the_model_layer() {
        let dir = std::env::temp_dir().join(format!("ds4-gencfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(GENERATION_CONFIG),
            r#"{"do_sample": false, "temperature": 1.0, "top_p": 0.95, "max_new_tokens": 2048}"#,
        )
        .unwrap();
        let s = model_source(&dir);
        assert_eq!(s.top_p, Some(0.95));
        assert_eq!(s.max_tokens, Some(2048));
        assert_eq!(s.min_p, None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unseeded_requests_draw_distinct_reproducible_seeds() {
        init_seeds(Some(7));
        let a = [next_seed(), next_seed()];
        init_seeds(Some(7));
        assert_eq!(a, [next_seed(), next_seed()]);
        assert_ne!(a[0], a[1]);
        assert_eq!(request_seed(0.0), 0);
        assert_eq!(request_seed(42.0), 42);
    }
}
