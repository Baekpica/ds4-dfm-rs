//! The audio-state gate: the LTX-2.5 audio VAE encoder against its golden.
//!
//! Usage: h3_ltx_audio_gate <audio-vae.safetensors> <pcm> --golden-latent FILE
//!        [--golden-mel FILE] [--relative N] [--frames N] [--dump FILE]
//!
//! `<pcm>` is the stage-1 soundtrack: raw little-endian interleaved F32 at
//! 32 kHz, the file `h3-rs --save-audio` writes and `h3-sol-rs --audio` reads.
//! The golden is what the reference's own Python produced
//! (`~/ds4-sol-h3/scratch/agold/make_golden.py`): the encoder's
//! `[1, 8, 130, 16]` float32 latent, channel-major, and optionally the
//! `[1, 2, 518, 64]` log mel that precedes it.
//!
//! Two checks, both loud:
//!
//! 1. the front end's log mel against the golden mel, per channel;
//! 2. the whole chain - mel, convolution stack, `double_z` means, the
//!    per-channel statistics - against the golden latent.
//!
//! The latent check is the gate: a missing or unreadable golden exits 3
//! rather than skipping, because a gate that cannot compare is not a gate.
//! The relative-RMS bar is 1e-4: the port is the same arithmetic as the
//! reference's float32 run, so only summation order differs.
//!
//! `--frames` prints the conformed latent the refiner actually takes
//! (`conformed[:, :, :frames]`, `runtime/stage2.py:88-89`); `--dump` writes it
//! as channel-major float32.
//!
//! Exit 3 when a check fails or a golden is missing.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::ltx_audio_vae::{
    log_mel, AudioLatent, Encoder, Mel, MEL_BINS, MEL_CHANNELS, SOURCE_RATE,
};
use ds4_sys::h3_dit::Gpu;

/// The bar the whole chain must stay inside: same expression, same F32
/// arithmetic, only the accumulation order inside a tap differs.
const RELATIVE: f64 = 1e-4;
/// The front end's own bar: a host transcription of torchaudio's resample,
/// STFT and slaney bank. Tighter than the chain's because it is the same
/// arithmetic down to the sample.
const MEL_RELATIVE: f64 = 1e-4;
/// The refiner's own audio frame count (`AUDIO_TOKENS`).
const REFINER_FRAMES: u32 = 126;

struct Options {
    weights: PathBuf,
    pcm: PathBuf,
    golden_latent: PathBuf,
    golden_mel: Option<PathBuf>,
    relative: f64,
    frames: u32,
    dump: Option<PathBuf>,
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();

    match run(&arguments) {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::from(1)
        }
    }
}

fn run(arguments: &[String]) -> Result<ExitCode, String> {
    let options = parse(arguments)?;
    let pcm = read_pcm(&options.pcm)?;
    let samples = pcm.len() / MEL_CHANNELS;
    println!(
        "pcm {} samples x {} channels ({:.3} s at {} Hz)",
        samples,
        MEL_CHANNELS,
        samples as f64 / SOURCE_RATE as f64,
        SOURCE_RATE
    );

    let started = Instant::now();
    let mel = log_mel(&pcm, MEL_CHANNELS, SOURCE_RATE).map_err(|error| error.to_string())?;
    let mel_time = started.elapsed();
    println!(
        "mel {} x {} x {} channels-last in {:.3?}",
        mel.frames, mel.bins, mel.channels, mel_time
    );

    let gpu = Gpu::create().map_err(|error| format!("device: {error}"))?;
    let encoder = Encoder::open(&gpu, &options.weights).map_err(|error| error.to_string())?;
    let (tensors, bytes) = encoder.loaded();
    println!("weights {tensors} tensors, {bytes} bytes");

    for shape in Encoder::shapes(mel.frames as u32) {
        println!(
            "stage {} x {} x {}",
            shape.channels, shape.frames, shape.bins
        );
    }

    let started = Instant::now();
    let latent = encoder
        .encode(&gpu, &mel)
        .map_err(|error| error.to_string())?;
    let encode_time = started.elapsed();
    println!(
        "latent {} x {} x {} in {:.3?}",
        latent.channels, latent.frames, latent.bins, encode_time
    );

    let mut failed = false;

    if let Some(path) = &options.golden_mel {
        let golden = read_f32(path, MEL_CHANNELS * mel.frames * MEL_BINS)?;
        let (relative, worst) = mel_error(&mel, &golden);
        println!(
            "front end: log-mel rel_rms {relative:.3e} max_abs {worst:.3e} against {}",
            path.display()
        );
        if relative > MEL_RELATIVE {
            failed = true;
        }
    } else {
        println!("front end: no golden mel named, the chain check below still covers it");
    }

    let golden = read_f32(&options.golden_latent, latent.values.len())?;
    let (relative, worst) = latent_error(&latent, &golden);
    println!(
        "chain: latent rel_rms {relative:.3e} max_abs {worst:.3e} against {}",
        options.golden_latent.display()
    );
    if relative > options.relative {
        failed = true;
    }

    let conformed = latent.conformed(options.frames);
    let values = &conformed.values;
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let spread = (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32)
        .sqrt();
    println!(
        "refiner state: {} x {} x {} mean {mean:.5} std {spread:.5}",
        conformed.channels, conformed.frames, conformed.bins
    );

    if let Some(path) = &options.dump {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))?;
        println!("wrote {}", path.display());
    }

    if failed {
        return Ok(ExitCode::from(3));
    }

    Ok(ExitCode::SUCCESS)
}

/// Per-channel relative RMS and the worst absolute error of the log mel.
fn mel_error(mel: &Mel, golden: &[f32]) -> (f64, f32) {
    let (mut numerator, mut denominator) = (0.0f64, 0.0f64);
    let mut worst = 0.0f32;

    for channel in 0..mel.channels {
        for frame in 0..mel.frames {
            for bin in 0..mel.bins {
                let mine = mel.values[(frame * mel.bins + bin) * mel.channels + channel];
                let reference = golden[(channel * mel.frames + frame) * mel.bins + bin];
                let difference = (mine - reference) as f64;
                numerator += difference * difference;
                denominator += (reference as f64) * (reference as f64);
                worst = worst.max(difference.abs() as f32);
            }
        }
    }

    ((numerator / denominator).sqrt(), worst)
}

fn latent_error(latent: &AudioLatent, golden: &[f32]) -> (f64, f32) {
    let (mut numerator, mut denominator) = (0.0f64, 0.0f64);
    let mut worst = 0.0f32;

    for (mine, reference) in latent.values.iter().zip(golden) {
        let difference = (*mine - *reference) as f64;
        numerator += difference * difference;
        denominator += (*reference as f64) * (*reference as f64);
        worst = worst.max(difference.abs() as f32);
    }

    ((numerator / denominator).sqrt(), worst)
}

fn read_pcm(path: &PathBuf) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() % (4 * MEL_CHANNELS) != 0 {
        return Err(format!(
            "{}: {} bytes is not whole stereo f32 frames",
            path.display(),
            bytes.len()
        ));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

fn read_f32(path: &PathBuf, elements: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != elements * 4 {
        return Err(format!(
            "{}: {} bytes, the graph needs {} f32 values ({} bytes)",
            path.display(),
            bytes.len(),
            elements,
            elements * 4
        ));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

fn parse(arguments: &[String]) -> Result<Options, String> {
    if arguments.len() < 2 {
        return Err(usage());
    }

    let mut options = Options {
        weights: PathBuf::from(&arguments[0]),
        pcm: PathBuf::from(&arguments[1]),
        golden_latent: PathBuf::new(),
        golden_mel: None,
        relative: RELATIVE,
        frames: REFINER_FRAMES,
        dump: None,
    };
    let mut rest = arguments[2..].iter();

    while let Some(flag) = rest.next() {
        let mut value = || {
            rest.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--golden-latent" => options.golden_latent = PathBuf::from(value()?),
            "--golden-mel" => options.golden_mel = Some(PathBuf::from(value()?)),
            "--relative" => {
                options.relative = value()?
                    .parse()
                    .map_err(|error| format!("--relative: {error}"))?
            }
            "--frames" => {
                options.frames = value()?
                    .parse()
                    .map_err(|error| format!("--frames: {error}"))?
            }
            "--dump" => options.dump = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown flag {other}\n\n{}", usage())),
        }
    }

    if options.golden_latent.as_os_str().is_empty() {
        return Err(format!(
            "no --golden-latent: the encoder gate compares against the reference's own latent\n\n{}",
            usage()
        ));
    }
    if !(options.relative > 0.0) {
        return Err("the relative bar must be positive".into());
    }

    Ok(options)
}

fn usage() -> String {
    "Usage: h3_ltx_audio_gate <audio-vae.safetensors> <pcm> --golden-latent FILE \
     [--golden-mel FILE] [--relative N] [--frames N] [--dump FILE]"
        .into()
}
