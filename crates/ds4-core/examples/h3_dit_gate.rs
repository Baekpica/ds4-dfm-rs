//! The E5 gate: the port's DiT forward against the reference's golden.
//!
//! Usage: h3_dit_gate <FL2VA/text_encoder> <tokenizer.json> <FL2VA/transformer>
//!                    <prompt> [--width W] [--height H] [--frames F]
//!                    [--steps N] [--step S] [--seed N]
//!                    [--expect-video HEX] [--expect-audio HEX]
//!                    [--expect-adaln BLOCK:FINAL] [--adaln-only]
//!
//! Prints the prompt ids, the text hash (the E4 gate's), the packed layout,
//! the seeded noise and the two velocities. With an `--expect-*` flag the
//! process exits 3 on a mismatch; `--adaln-only` stops after the AdaLN
//! projections, which is the cheapest gate in the unit.
//!
//! The goldens this gate was written against, recorded on the Spark with the
//! reference engine (`tools/minimax_h3/dit_probe.c`), seed 1, step 0:
//!
//! 256x256, 5 frames, two evaluations:
//!
//!   TOKENS 32 2518 38835 11435 1526 11794
//!   TEXT tokens=6 width=5120 fnv=1d9d71e8dd8544d6
//!   LAYOUT rows=150 segments=3 positions=728dafc2ad799386
//!   SIGMAS steps=2 video=3f800000.3f6c4ec5 audio=3f800000.3f400000
//!   ADALN rows=3 block0=86f97b32732707a2 final=e61bd4d3ae8b5aa9
//!   NOISE seed=1 video=12288=5691802bdd5b0074 audio=512=094d71985b0c77d7
//!   FORWARD video=ee802a99530734e2 audio=6737524c48b8f549
//!   TIME seconds=0.249 (page cache warm) after LOAD seconds=30.143
//!
//! 512x256, 22 frames, four evaluations (`--width 512 --height 256 --frames 22
//! --steps 4`):
//!
//!   LAYOUT rows=976 segments=3 positions=1f72c77e7cfd2a0b
//!   SIGMAS steps=4 video=3f800000.3f7914c2.3f6c4ec5.3f4ccccd
//!          audio=3f800000.3f666666.3f400000.3f000000
//!   NOISE seed=1 video=86016=a7ad6c7fffd4883c audio=2368=a6f0828e5dd3268f
//!   FORWARD video=19ded4b06fdf2212 audio=e485560aea6da48d
//!   TIME seconds=1.735 after LOAD seconds=20.436

use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::dit::{self, Dit};
use ds4_core::minimax_h3::host::{latent_canvas, temporal, Rng, Schedule};
use ds4_core::minimax_h3::sequence::{self, Spec};
use ds4_core::minimax_h3::text_forward;
use ds4_core::minimax_h3::tokenizer::{EmptyPrompt, Tokenizer};

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
    if options.adaln_only {
        if positional.len() != 1 {
            return Err(usage("--adaln-only takes the transformer directory only"));
        }
        return adaln_gate(&positional[0], &options);
    }
    if positional.len() != 4 {
        return Err(usage("expected four positional arguments"));
    }

    let encoder = std::path::Path::new(&positional[0]);
    let tokenizer_path = std::path::Path::new(&positional[1]);
    let transformer = std::path::Path::new(&positional[2]);
    let prompt = &positional[3];

    let tokenizer =
        Tokenizer::load(tokenizer_path).map_err(|error| format!("tokenizer: {}", error.token()))?;
    let ids = tokenizer
        .encode(prompt, EmptyPrompt::Pad)
        .map_err(|error| format!("tokenizer: {}", error.token()))?;

    println!(
        "TOKENS {}",
        ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
    );

    let start = Instant::now();
    let text = text_forward::encode(encoder, &ids).map_err(|error| error.token())?;
    println!(
        "TEXT tokens={} width={} fnv={:016x} seconds={:.3}",
        text.tokens(),
        text.width(),
        text.fnv(),
        start.elapsed().as_secs_f64()
    );

    let temporal = temporal(options.frames);
    let (latent_w, latent_h) = latent_canvas(options.width, options.height);
    let spec = Spec {
        text_len: text.tokens(),
        latent_t: temporal.video_t,
        latent_h,
        latent_w,
        audio_t: temporal.audio_t,
        frame_count: temporal.frame_count,
        keyframes: &[],
    };
    let sequence = sequence::build(&spec).map_err(|error| format!("layout: {error:?}"))?;
    println!(
        "LAYOUT rows={} segments={} positions={:016x} cond={}+{} target={}+{} latent={}x{}x{} audio_t={}",
        sequence.rows(),
        sequence.segments.len(),
        positions_hash(&sequence),
        sequence.img_cond_rows,
        sequence.audio_cond_rows,
        sequence.img_target_rows,
        sequence.audio_target_rows,
        temporal.video_t,
        latent_h,
        latent_w,
        temporal.audio_t
    );

    let schedule =
        Schedule::serving(options.steps).ok_or_else(|| usage("--steps is outside 2..=1000"))?;
    println!("{}", sigmas_line(&schedule));

    let start = Instant::now();
    let model = Dit::load(
        transformer,
        sequence,
        schedule.clone(),
        &text,
        options.rope_scale,
    )
    .map_err(|error| error.to_string())?;
    println!("LOAD seconds={:.3}", start.elapsed().as_secs_f64());

    let mut video_noise = vec![0.0f32; model.video_elements()];
    let mut audio_noise = vec![0.0f32; model.audio_elements()];
    Rng::seed(options.seed).fill_normal(&mut video_noise);
    Rng::seed(options.seed).fill_normal(&mut audio_noise);
    println!(
        "NOISE seed={} video={}={:016x} audio={}={:016x} video_edge={:.9}/{:.9} audio_edge={:.9}/{:.9}",
        options.seed,
        video_noise.len(),
        fnv_f32(&video_noise),
        audio_noise.len(),
        fnv_f32(&audio_noise),
        video_noise[0],
        video_noise[video_noise.len() - 1],
        audio_noise[0],
        audio_noise[audio_noise.len() - 1]
    );

    let start = Instant::now();
    let (video, audio) = model
        .forward(options.step, &video_noise, &audio_noise)
        .map_err(|error| error.to_string())?;
    let seconds = start.elapsed().as_secs_f64();

    println!(
        "FORWARD video={:016x} audio={:016x} video_edge={:.9}/{:.9} audio_edge={:.9}/{:.9}",
        fnv_f32(&video),
        fnv_f32(&audio),
        video[0],
        video[video.len() - 1],
        audio[0],
        audio[audio.len() - 1]
    );
    println!("TIME seconds={seconds:.3}");

    if let Some(expect) = options.expect_video {
        if fnv_f32(&video) != expect {
            return Ok(ExitCode::from(3));
        }
    }
    if let Some(expect) = options.expect_audio {
        if fnv_f32(&audio) != expect {
            return Ok(ExitCode::from(3));
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// The AdaLN-only gate: `h3_dit_schedule_precompute` on its own, hashed as the
/// reference stores it.
fn adaln_gate(directory: &str, options: &Options) -> Result<ExitCode, String> {
    let schedule =
        Schedule::serving(options.steps).ok_or_else(|| usage("--steps is outside 2..=1000"))?;
    println!("{}", sigmas_line(&schedule));

    let start = Instant::now();
    let hashes = dit::adaln_hashes(std::path::Path::new(directory), schedule)
        .map_err(|error| error.to_string())?;
    println!(
        "ADALN embed={:016x} block0={:016x} final={:016x} seconds={:.3}",
        hashes.embed,
        hashes.block,
        hashes.final_layer,
        start.elapsed().as_secs_f64()
    );

    if let Some((expect_block, expect_final)) = options.expect_adaln {
        if hashes.block != expect_block || hashes.final_layer != expect_final {
            return Ok(ExitCode::from(3));
        }
    }

    Ok(ExitCode::SUCCESS)
}

fn sigmas_line(schedule: &Schedule) -> String {
    let video = schedule
        .video
        .iter()
        .take(schedule.steps)
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(".");
    let audio = schedule
        .audio
        .iter()
        .take(schedule.steps)
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(".");
    format!(
        "SIGMAS steps={} video={video} audio={audio}",
        schedule.steps
    )
}

/// `h3_dit.c:956-968`: FNV over the `{t, h, w}` doubles in that order.
fn positions_hash(sequence: &sequence::Sequence) -> u64 {
    let mut bytes = Vec::with_capacity(sequence.rows() * 24);
    for position in &sequence.positions {
        for value in [position.t, position.h, position.w] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    text_forward::fnv1a(&bytes)
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
    width: i32,
    height: i32,
    frames: i32,
    steps: usize,
    step: usize,
    seed: u64,
    rope_scale: f32,
    expect_video: Option<u64>,
    expect_audio: Option<u64>,
    expect_adaln: Option<(u64, u64)>,
    adaln_only: bool,
}

fn split(args: &[String]) -> Result<(Vec<String>, Options), String> {
    let mut positional = Vec::new();
    let mut options = Options {
        width: 256,
        height: 256,
        frames: 5,
        steps: 2,
        step: 0,
        seed: 1,
        rope_scale: 1.0,
        expect_video: None,
        expect_audio: None,
        expect_adaln: None,
        adaln_only: false,
    };
    let mut index = 0;

    while index < args.len() {
        let argument = &args[index];
        let value = || {
            args.get(index + 1)
                .ok_or_else(|| usage(&format!("{argument} needs a value")))
        };

        match argument.as_str() {
            "--width" => options.width = number(value()?)?,
            "--height" => options.height = number(value()?)?,
            "--frames" => options.frames = number(value()?)?,
            "--steps" => options.steps = number(value()?)?,
            "--step" => options.step = number(value()?)?,
            "--seed" => options.seed = number(value()?)?,
            "--rope-scale" => options.rope_scale = number(value()?)?,
            "--expect-video" => {
                options.expect_video = Some(
                    parse_hash(value()?)
                        .ok_or_else(|| usage("--expect-video needs 16 hex digits"))?,
                );
            }
            "--expect-audio" => {
                options.expect_audio = Some(
                    parse_hash(value()?)
                        .ok_or_else(|| usage("--expect-audio needs 16 hex digits"))?,
                );
            }
            "--expect-adaln" => {
                let text = value()?;
                let (block, final_layer) = text
                    .split_once(':')
                    .ok_or_else(|| usage("--expect-adaln needs BLOCK:FINAL"))?;
                options.expect_adaln = Some((
                    parse_hash(block).ok_or_else(|| usage("--expect-adaln needs 16 hex digits"))?,
                    parse_hash(final_layer)
                        .ok_or_else(|| usage("--expect-adaln needs 16 hex digits"))?,
                ));
            }
            "--adaln-only" => {
                options.adaln_only = true;
                index += 1;
                continue;
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
        "{detail}\nusage: h3_dit_gate <FL2VA/text_encoder> <tokenizer.json> \
         <FL2VA/transformer> <prompt> [--width W] [--height H] [--frames F] \
         [--steps N] [--step S] [--seed N] [--rope-scale X] [--expect-video HEX] \
         [--expect-audio HEX] [--expect-adaln BLOCK:FINAL]\n       \
         h3_dit_gate <FL2VA/transformer> --adaln-only [--steps N] \
         [--expect-adaln BLOCK:FINAL]"
    )
}
