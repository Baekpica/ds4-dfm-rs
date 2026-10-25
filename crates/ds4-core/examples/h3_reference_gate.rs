//! Prepare ordered references and report their conditioning geometry.
//!
//! Usage: h3_reference_gate <model_dir> <image> <video> <soundtrack>
//!        [--frames N] [--seed N] [--policy match|max]
//!
//! Three references are prepared through `minimax_h3::reference::prepare`,
//! the three kinds the reference distinguishes:
//!
//! 1. `image`: an image reference at the requested sizing policy.
//! 2. `video`: a video reference whose own track is its soundtrack
//!    (`include_embedded_audio`, `h3.c:1133-1136`).
//! 3. `video+audio`: a video reference with `<soundtrack>` as its separate
//!    audio (`h3.c:1131-1133`).
//!
//! Each prints one `REFERENCE` line: the media canvas, the decoded frames, the
//! latent geometry, the row counts and an FNV-1a over the rows, the audio
//! latent length and audio rows, the vision block count and timestamps.
//! `TOTAL` carries the audio samples of the set, which the reference caps at
//! 15 s.
//!
//! This gate is port-side: the primitives under it (canvas policy, video VAE
//! encoder, audio VAE encoder, media readers) each have their own two-sided
//! gate. The two-sided proof of the composition is P14.6's render parity.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::partition::Partition;
use ds4_core::minimax_h3::reference::{
    prepare, ImageSize, Kind, Paths, Prepared, Reference, MAX_AUDIO_SAMPLES, MAX_REFERENCES,
};
use ds4_core::minimax_h3::text_forward;

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<String> = Vec::new();
    let mut frames = 107usize;
    let mut seed = 42u64;
    let mut policy = ImageSize::Match;

    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--frames" => {
                index += 1;
                match arguments.get(index).and_then(|value| value.parse().ok()) {
                    Some(value) => frames = value,
                    None => return usage("--frames needs a count"),
                }
            }
            "--seed" => {
                index += 1;
                match arguments.get(index).and_then(|value| value.parse().ok()) {
                    Some(value) => seed = value,
                    None => return usage("--seed needs a number"),
                }
            }
            "--policy" => {
                index += 1;
                match arguments.get(index).map(String::as_str) {
                    Some("match") => policy = ImageSize::Match,
                    Some("max") => policy = ImageSize::Max,
                    _ => return usage("--policy needs match or max"),
                }
            }
            "--help" | "-h" => return usage(""),
            value if value.starts_with("--") => return usage(&format!("unknown flag {value}")),
            value => positional.push(value.to_string()),
        }
        index += 1;
    }

    if positional.len() != 4 {
        return usage("a model directory, an image, a video and a soundtrack are required");
    }

    let model_dir = PathBuf::from(&positional[0]);
    let image = PathBuf::from(&positional[1]);
    let video = PathBuf::from(&positional[2]);
    let soundtrack = PathBuf::from(&positional[3]);

    let video_vae = Partition::Fl2va.component(&model_dir, "video_vae/source");
    let audio_vae = Partition::Fl2va.component(&model_dir, "audio_vae");
    let paths = Paths {
        video_vae: &video_vae,
        audio_vae: &audio_vae,
    };

    let references = [
        Reference {
            kind: Kind::Image,
            path: image,
            audio_path: None,
            include_embedded_audio: false,
        },
        Reference {
            kind: Kind::Video,
            path: video.clone(),
            audio_path: None,
            include_embedded_audio: true,
        },
        Reference {
            kind: Kind::VideoAudio,
            path: video,
            audio_path: Some(soundtrack),
            include_embedded_audio: false,
        },
    ];
    if references.len() > MAX_REFERENCES {
        eprintln!("too many references");
        return ExitCode::FAILURE;
    }

    let render = (1344u32, 768u32);
    let mut audio_total = 0usize;
    let started = Instant::now();
    for (position, reference) in references.iter().enumerate() {
        let prepared = match prepare(&paths, reference, policy, render, frames, seed, position) {
            Ok(prepared) => prepared,
            Err(error) => {
                eprintln!("{}", error.message());
                return ExitCode::FAILURE;
            }
        };
        report(position, reference, &prepared);
        audio_total += prepared.audio_samples;
    }

    let within_budget = audio_total <= MAX_AUDIO_SAMPLES;
    println!(
        "TOTAL\treferences={}\taudio_samples={audio_total}\twithin_15s={}\tseconds={:.3}",
        references.len(),
        within_budget,
        started.elapsed().as_secs_f64()
    );
    match within_budget {
        true => ExitCode::SUCCESS,
        false => {
            eprintln!("ordered reference audio exceeds 15 seconds in total");
            ExitCode::FAILURE
        }
    }
}

fn report(position: usize, reference: &Reference, prepared: &Prepared) {
    let timestamps = prepared
        .timestamps
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "REFERENCE\tindex={}\tkind={}\twidth={}\theight={}\tframes={}\tlatent={}x{}x{}\t\
         row_elements={}\trows_fnv={:016x}\taudio_t={}\taudio_rows={}\taudio_fnv={:016x}\t\
         samples={}\tvision_blocks={}\ttimestamps={}",
        position + 1,
        kind_name(reference.kind),
        prepared.width,
        prepared.height,
        prepared.frames,
        prepared.latent_t,
        prepared.latent_h,
        prepared.latent_w,
        prepared.video_rows.len(),
        fnv(&prepared.video_rows),
        prepared.audio_t,
        prepared.audio_rows.len(),
        fnv(&prepared.audio_rows),
        prepared.audio_samples,
        prepared.vision_blocks,
        timestamps
    );
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Image => "image",
        Kind::Video => "video",
        Kind::Audio => "audio",
        Kind::VideoAudio => "video+audio",
    }
}

/// FNV-1a over the rows' little-endian F32 bytes, the gate family's
/// convention.
fn fnv(values: &[f32]) -> u64 {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    text_forward::fnv1a(&bytes)
}

fn usage(detail: &str) -> ExitCode {
    if !detail.is_empty() {
        eprintln!("{detail}");
    }
    eprintln!(
        "usage: h3_reference_gate <model_dir> <image> <video> <soundtrack>\n\
         \x20      [--frames N] [--seed N] [--policy match|max]"
    );
    ExitCode::from(2)
}
