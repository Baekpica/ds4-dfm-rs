//! Inkling's fixed 16 kHz waveform to 80-band discrete mel features.

use std::io::Cursor;
use std::sync::{Arc, OnceLock};

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::{Error, Result};

const SAMPLE_RATE: usize = 16000;
const HOP: usize = 800;
const FFT_SIZE: usize = 1600;
const FREQ_BINS: usize = FFT_SIZE / 2 + 1;
const MEL_BINS: usize = 80;
const LEVELS: usize = 16;
const MAX_FRAMES: usize = 8192;
const MAX_ENCODED: usize = 32 * 1024 * 1024;
const MAX_CHANNELS: u16 = 8;

pub(crate) struct AudioCodes {
    pub(crate) codes: Vec<i32>,
    pub(crate) frames: u32,
}

struct AudioPlan {
    fft: Arc<dyn Fft<f32>>,
    window: [f32; FFT_SIZE],
    filters: Vec<Vec<(usize, f32)>>,
}

fn audio_error(message: &str) -> Error {
    Error {
        code: 1,
        message: message.into(),
    }
}

fn frame_count(samples: usize, budget: u32) -> Result<usize> {
    if samples == 0 || samples > MAX_FRAMES * HOP {
        return Err(audio_error(
            "Inkling audio length is outside the supported range",
        ));
    }
    let frames = samples.div_ceil(HOP);
    if frames > budget as usize {
        return Err(audio_error("Inkling audio exceeds the token budget"));
    }
    Ok(frames)
}

fn open_wave(data: &[u8]) -> Result<hound::WavReader<Cursor<&[u8]>>> {
    if data.is_empty() || data.len() > MAX_ENCODED {
        return Err(audio_error("Inkling audio exceeds the encoded byte limit"));
    }
    let reader = hound::WavReader::new(Cursor::new(data))
        .map_err(|e| audio_error(&format!("invalid Inkling WAV: {e}")))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE as u32 || spec.channels > MAX_CHANNELS {
        return Err(audio_error(
            "Inkling requires 16 kHz WAV with 1 to 8 channels",
        ));
    }
    if !matches!(spec.bits_per_sample, 8 | 16 | 24 | 32) {
        return Err(audio_error("unsupported Inkling WAV sample depth"));
    }
    Ok(reader)
}

pub(crate) fn probe_audio(data: &[u8], budget: u32) -> Result<u32> {
    let reader = open_wave(data)?;
    Ok(frame_count(reader.duration() as usize, budget)? as u32)
}

fn read_mono(
    mut samples: impl Iterator<Item = hound::Result<f32>>,
    channels: u16,
    count: usize,
) -> Result<Vec<f32>> {
    let mut wave = Vec::new();
    wave.try_reserve_exact(count)
        .map_err(|_| audio_error("Inkling waveform allocation failed"))?;
    for _ in 0..count {
        let mut sum = 0.0;
        for _ in 0..channels {
            let value = samples
                .next()
                .ok_or_else(|| audio_error("truncated Inkling WAV"))?
                .map_err(|e| audio_error(&format!("invalid Inkling WAV sample: {e}")))?;
            if !value.is_finite() {
                return Err(audio_error("Inkling audio contains nonfinite samples"));
            }
            sum += value;
        }
        wave.push(sum / channels as f32);
    }
    Ok(wave)
}

pub(crate) fn prepare_audio(data: &[u8], budget: u32) -> Result<AudioCodes> {
    let mut reader = open_wave(data)?;
    let count = reader.duration() as usize;
    let frames = frame_count(count, budget)? as u32;
    let spec = reader.spec();
    // Match source channel-mean downmix. Rate conversion is intentionally
    // rejected until a resampler has its own numerical contract.
    let wave = match spec.sample_format {
        hound::SampleFormat::Float => read_mono(reader.samples::<f32>(), spec.channels, count)?,
        hound::SampleFormat::Int => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            read_mono(
                reader
                    .samples::<i32>()
                    .map(|sample| sample.map(|v| v as f32 / scale)),
                spec.channels,
                count,
            )?
        }
    };
    let codes = extract_mel(&wave, frames)?
        .into_iter()
        .map(quantize_mel)
        .collect();
    Ok(AudioCodes { codes, frames })
}

fn audio_plan() -> &'static AudioPlan {
    static PLAN: OnceLock<AudioPlan> = OnceLock::new();
    PLAN.get_or_init(|| {
        // HF constructs the Slaney triangles in FP64, then casts to FP32.
        let log_step = 6.4f64.ln() / 27.0;
        let mel_max = 15.0 + (8.0f64).ln() / log_step;
        let points: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| {
                let mel = i as f64 * (mel_max / (MEL_BINS + 1) as f64);
                if mel >= 15.0 {
                    1000.0 * (log_step * (mel - 15.0)).exp()
                } else {
                    200.0 * mel / 3.0
                }
            })
            .collect();
        let filters = (0..MEL_BINS)
            .map(|m| {
                (0..FREQ_BINS)
                    .filter_map(|k| {
                        let hz = k as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
                        let down = (hz - points[m]) / (points[m + 1] - points[m]);
                        let up = (points[m + 2] - hz) / (points[m + 2] - points[m + 1]);
                        let weight =
                            (down.min(up).max(0.0) * (2.0 / (points[m + 2] - points[m]))) as f32;
                        (weight > 0.0).then_some((k, weight))
                    })
                    .collect()
            })
            .collect();
        let step = (2.0 * std::f64::consts::PI / FFT_SIZE as f64) as f32;
        AudioPlan {
            fft: FftPlanner::new().plan_fft_forward(FFT_SIZE),
            window: std::array::from_fn(|i| -0.5 * (i as f32 * step).cos() + 0.5),
            filters,
        }
    })
}

fn extract_mel(wave: &[f32], budget: u32) -> Result<Vec<f32>> {
    let frames = frame_count(wave.len(), budget)?;
    if wave.iter().any(|v| !v.is_finite()) {
        return Err(audio_error("Inkling audio contains nonfinite samples"));
    }
    let plan = audio_plan();
    let mut fft = vec![Complex32::default(); FFT_SIZE];
    let mut scratch = vec![Complex32::default(); plan.fft.get_inplace_scratch_len()];
    let mut magnitudes = [0.0f32; FREQ_BINS];
    let mut mel = Vec::with_capacity(frames * MEL_BINS);
    for frame in 0..frames {
        // center=false, left pad 800, right pad to the next hop boundary.
        // Index padding directly so long clips need no duplicate waveform.
        for (i, bin) in fft.iter_mut().enumerate() {
            let sample = (frame * HOP + i)
                .checked_sub(FFT_SIZE - HOP)
                .and_then(|index| wave.get(index))
                .copied()
                .unwrap_or(0.0);
            *bin = Complex32::new(sample * plan.window[i], 0.0);
        }
        plan.fft.process_with_scratch(&mut fft, &mut scratch);
        for (magnitude, bin) in magnitudes.iter_mut().zip(&fft) {
            let power = bin.re * bin.re + bin.im * bin.im;
            if !power.is_finite() {
                return Err(audio_error("Inkling audio spectrum is nonfinite"));
            }
            *magnitude = power.max(1e-10).sqrt();
        }
        for filter in &plan.filters {
            let energy: f32 = filter
                .iter()
                .map(|(k, weight)| magnitudes[*k] * weight)
                .sum();
            mel.push(energy.max(1e-10).log10());
        }
    }
    Ok(mel)
}

fn quantize_mel(value: f32) -> i32 {
    let value = value.clamp(-7.0, 2.0);
    let mut best = 0;
    let mut distance = f32::INFINITY;
    for code in 0..LEVELS {
        // The source builds linspace in FP64 before its FP32 distance/argmin.
        let center = (-7.0 + code as f64 * (9.0 / (LEVELS - 1) as f64)) as f32;
        let next = (value - center).abs();
        if next < distance {
            distance = next;
            best = code as i32;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn reference() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/inkling/audio-vectors.json"
        ))
        .unwrap()
    }

    #[test]
    fn audio_source_features() {
        let reference = reference();
        for case in reference["cases"].as_array().unwrap() {
            let count = case["samples"].as_u64().unwrap() as usize;
            let wave: Vec<f32> = match case["waveform"].as_array() {
                Some(values) => values.iter().map(|v| v.as_f64().unwrap() as f32).collect(),
                None => (0..count)
                    .map(|i| {
                        (((i as u64 * 1664525 + 1013904223) & 0xffff) as i32 - 32768) as f32
                            / 131072.0
                    })
                    .collect(),
            };
            let mel = extract_mel(&wave, count.div_ceil(800) as u32).unwrap();
            let expected: Vec<f32> = case["log_mel"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(mel.len(), expected.len());
            let mut max_error = 0.0f32;
            let mut max_linear = 0.0f32;
            for (got, want) in mel.iter().zip(&expected) {
                max_error = max_error.max((got - want).abs());
                let got_energy = 10.0f32.powf(*got);
                let want_energy = 10.0f32.powf(*want);
                max_linear = max_linear.max((got_energy - want_energy).abs());
                // FFT rounding dominates nearly silent tonal bands. Bound
                // the pre-log energy; require exact downstream dMel codes.
                assert!((got_energy - want_energy).abs() < 2e-7 + 3e-6 * want_energy);
            }
            let codes: Vec<i32> = mel.into_iter().map(quantize_mel).collect();
            let expected: Vec<i32> = case["codes"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            assert_eq!(codes, expected, "{} {count}", case["name"]);
            eprintln!(
                "{} {count}: codes exact, max energy error {max_linear}, log error {max_error}",
                case["name"]
            );
        }
    }

    #[test]
    fn audio_bins_and_limits() {
        let reference = reference();
        let mut window_error = 0.0f32;
        for (got, want) in audio_plan()
            .window
            .iter()
            .zip(reference["window"].as_array().unwrap())
        {
            window_error = window_error.max((got - want.as_f64().unwrap() as f32).abs());
        }
        assert!(window_error <= f32::EPSILON);
        for (value, code) in reference["quant_values"]
            .as_array()
            .unwrap()
            .iter()
            .zip(reference["quant_codes"].as_array().unwrap())
        {
            assert_eq!(
                quantize_mel(value.as_f64().unwrap() as f32),
                code.as_i64().unwrap() as i32
            );
        }
        assert!(extract_mel(&[], 1).is_err());
        assert!(extract_mel(&[0.0], 0).is_err());
        assert!(extract_mel(&[f32::NAN], 1).is_err());
        assert!(extract_mel(&[f32::INFINITY], 1).is_err());
        assert!(extract_mel(&[0.0; 801], 1).is_err());
        assert!(frame_count(MAX_FRAMES * HOP + 1, u32::MAX).is_err());
        assert_eq!(frame_count(MAX_FRAMES * HOP, u32::MAX).unwrap(), MAX_FRAMES);
    }

    fn wave_bytes(channels: u16, rate: u32, bits: u16, format: u16, data: &[u8]) -> Vec<u8> {
        let block = channels * (bits / 8);
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt \x10\0\0\0");
        wav.extend_from_slice(&format.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * block as u32).to_le_bytes());
        wav.extend_from_slice(&block.to_le_bytes());
        wav.extend_from_slice(&bits.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(data);
        wav
    }

    #[test]
    fn audio_wave_decode() {
        const PCM: u16 = 1;
        const IEEE_FLOAT: u16 = 3;
        let samples = [0i16, 16384, -8192].repeat(400);
        let mono: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let pcm8: Vec<u8> = samples
            .iter()
            .map(|v| ((*v as i32 >> 8) + 128) as u8)
            .collect();
        let pcm24: Vec<u8> = samples
            .iter()
            .flat_map(|v| ((*v as i32) << 8).to_le_bytes()[..3].to_vec())
            .collect();
        let pcm32: Vec<u8> = samples
            .iter()
            .flat_map(|v| ((*v as i32) << 16).to_le_bytes())
            .collect();
        let stereo: Vec<u8> = samples
            .iter()
            .flat_map(|v| [*v, *v])
            .flat_map(i16::to_le_bytes)
            .collect();
        let wave: Vec<f32> = samples.iter().map(|v| *v as f32 / 32768.0).collect();
        let floats: Vec<u8> = wave.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mixed: Vec<u8> = wave
            .iter()
            .flat_map(|v| [*v * 2.0, 0.0])
            .flat_map(f32::to_le_bytes)
            .collect();
        let expected: Vec<i32> = extract_mel(&wave, 2)
            .unwrap()
            .into_iter()
            .map(quantize_mel)
            .collect();
        for wav in [
            wave_bytes(1, 16000, 8, PCM, &pcm8),
            wave_bytes(1, 16000, 16, PCM, &mono),
            wave_bytes(1, 16000, 24, PCM, &pcm24),
            wave_bytes(1, 16000, 32, PCM, &pcm32),
            wave_bytes(2, 16000, 16, PCM, &stereo),
            wave_bytes(1, 16000, 32, IEEE_FLOAT, &floats),
            wave_bytes(2, 16000, 32, IEEE_FLOAT, &mixed),
        ] {
            assert_eq!(probe_audio(&wav, 2).unwrap(), 2);
            let prepared = prepare_audio(&wav, 2).unwrap();
            assert_eq!(prepared.frames, 2);
            assert_eq!(prepared.codes, expected);
            assert!(probe_audio(&wav, 1).is_err());
            assert!(prepare_audio(&wav[..wav.len() - 1], 2).is_err());
        }
        let eight: Vec<u8> = samples
            .iter()
            .flat_map(|v| [*v, 0, *v, 0, *v, 0, *v, 0])
            .flat_map(i16::to_le_bytes)
            .collect();
        let half: Vec<f32> = wave.iter().map(|v| *v / 2.0).collect();
        let expected: Vec<i32> = extract_mel(&half, 2)
            .unwrap()
            .into_iter()
            .map(quantize_mel)
            .collect();
        assert_eq!(
            prepare_audio(&wave_bytes(8, 16000, 16, PCM, &eight), 2)
                .unwrap()
                .codes,
            expected
        );
        assert!(probe_audio(&wave_bytes(1, 48000, 16, PCM, &mono), 2).is_err());
        assert!(prepare_audio(
            &wave_bytes(1, 16000, 32, IEEE_FLOAT, &f32::NAN.to_le_bytes()),
            1
        )
        .is_err());
        assert!(prepare_audio(&wave_bytes(1, 16000, 16, PCM, &[]), 1).is_err());
        assert!(probe_audio(b"not a wave", 1).is_err());
    }
}
