//! The P14.1 gate: the port's AudioVAE encoder against the reference's own
//! latent, hashed byte for byte.
//!
//! Usage: h3_audio_encoder_gate <FL2VA/audio_vae> <media>
//!        [--samples N] [--expect HEX]
//!
//! `<media>` is decoded by the same reader the reference uses
//! (`h3_ffmpeg.c:392-521`): first audio stream, stereo 32 kHz F32, truncated
//! at `--samples` (default 64000, the reference test's own length,
//! `tests/test_real_audio_encoder.c:20`). Both sides therefore see the same
//! channel-major PCM for the same file and limit.
//!
//! Output, in order:
//!
//!   PCM samples= values= fnv= first= last=
//!   LATENT channels= stereo= t= values= fnv= first= last=
//!   SECONDS read= encode= total=
//!
//! The hash covers the latent as `h3_audio_latent.values` carries it, which
//! is channel-major `[32,2,t]` after the released inverse normalization
//! (`h3_audio_vae.c:1253-1268`). It is the same FNV-1a convention as the
//! rest of the h3 gates, so the numbers printed here and by
//! `tools/minimax_h3/audio_encoder_probe.c` are directly comparable.
//!
//! With `--expect` the process exits 3 on a mismatch.

use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::media;
use ds4_core::minimax_h3::text_forward;
use ds4_core::minimax_h3::vae_audio;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match run(&args) {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let (positional, options) = split(args)?;
    if positional.len() != 2 {
        return Err(usage("expected two positional arguments"));
    }

    let directory = std::path::Path::new(&positional[0]);
    let media_path = std::path::Path::new(&positional[1]);

    let started = Instant::now();
    let pcm = media::read_audio_f32(media_path, options.samples, true)
        .map_err(|error| error.detail().to_string())?;
    let read_seconds = started.elapsed().as_secs_f64();
    let samples = pcm.len() / media::CHANNELS;

    println!(
        "PCM samples={} values={} fnv={:016x} first={:.9} last={:.9}",
        samples,
        pcm.len(),
        fnv_f32(&pcm),
        pcm[0],
        pcm[pcm.len() - 1]
    );

    let started = Instant::now();
    let latent = vae_audio::encode(directory, &pcm, samples).map_err(|error| error.to_string())?;
    let encode_seconds = started.elapsed().as_secs_f64();

    let hash = fnv_f32(&latent.values);
    println!(
        "LATENT channels={} stereo={} t={} values={} fnv={:016x} first={:.9} last={:.9}",
        latent.channels,
        latent.stereo,
        latent.length,
        latent.values.len(),
        hash,
        latent.values[0],
        latent.values[latent.values.len() - 1]
    );
    println!(
        "SECONDS read={read_seconds:.3} encode={encode_seconds:.3} total={:.3}",
        read_seconds + encode_seconds
    );

    if let Some(expect) = options.expect {
        if hash != expect {
            return Ok(ExitCode::from(3));
        }
    }

    Ok(ExitCode::SUCCESS)
}

fn fnv_f32(values: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    text_forward::fnv1a(&bytes)
}

fn parse_hash(text: &str) -> Option<u64> {
    if text.len() != 16 {
        return None;
    }
    u64::from_str_radix(text, 16).ok()
}

#[derive(Debug)]
struct Options {
    samples: u32,
    expect: Option<u64>,
}

fn split(args: &[String]) -> Result<(Vec<String>, Options), String> {
    let mut positional = Vec::new();
    let mut options = Options {
        samples: 64000,
        expect: None,
    };
    let mut index = 0;

    while index < args.len() {
        let argument = &args[index];
        let value = || {
            args.get(index + 1)
                .ok_or_else(|| usage(&format!("{argument} needs a value")))
        };

        match argument.as_str() {
            "--samples" => options.samples = number(value()?)?,
            "--expect" => {
                options.expect = Some(
                    parse_hash(value()?).ok_or_else(|| usage("--expect needs 16 hex digits"))?,
                );
            }
            _ if argument.starts_with("--") => {
                return Err(usage(&format!("unknown option {argument}")));
            }
            _ => {
                positional.push(argument.clone());
                index += 1;
                continue;
            }
        }
        index += 2;
    }

    Ok((positional, options))
}

fn number<T: std::str::FromStr>(text: &str) -> Result<T, String> {
    text.parse::<T>()
        .map_err(|_| usage(&format!("{text} is not a number")))
}

fn usage(detail: &str) -> String {
    format!(
        "{detail}\nusage: h3_audio_encoder_gate <FL2VA/audio_vae> <media> \
         [--samples N] [--expect HEX]"
    )
}
