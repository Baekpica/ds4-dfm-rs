//! Does the resident engine pay the prompt encode and the DiT load once?
//!
//! Usage: h3_engine_gate [--width N --height N --frames N --steps N]
//!        [--draft [--refine-sigma S] --upscaler DIR]
//!
//! One `EngineHandle` renders the same request twice and prints both phase
//! reports. The second render must hit the prompt embedding and the prepared
//! DiT, and both renders must produce the same media hashes: residency is
//! worth nothing if it changes the clip.
//!
//! `--draft` renders the reference's two-stage route instead
//! (`harness/workflow.py:50`): a draft at 0.7 of the canvas, the latent
//! upscaled, one refine step at 0.72 with the draft's audio frozen. That arm
//! asserts what the route is about - the audio the refine conditioned on is the
//! audio the decoder received (`draft_audio_hash == audio_hash`, and the
//! engine's own `audio_frozen`) - and reports the split per stage. Its repeat
//! cannot be a prepared hit: the engine holds one prepared graph and the two
//! stages are two canvases, so the claim it makes on the second render is the
//! prompt encode, not the load.
//!
//! Live gate: it needs the FastH3 checkpoint and a GPU. The checkpoint is
//! named by the environment, never by a path in this file: `H3_MODEL` for the
//! model directory, or `H3_ARTIFACTS` plus `H3_MODEL_NAME`. Unset, or absent
//! on disk, it exits 3, which is the convention `make test-h3` reads. The
//! cache-key rules themselves are covered model-free by the `engine` module's
//! own tests.

use std::path::PathBuf;
use std::process::ExitCode;

use ds4_core::minimax_h3::dit::{Attention, VaeAttention, VaeGemm};
use ds4_core::minimax_h3::engine::{Draft, EngineHandle, RenderReport, RenderRequest};
use ds4_core::minimax_h3::render::Encode;
use ds4_core::minimax_h3::text_forward::Encoder;

/// The release's stage-1 canvas is 672x384x124; the gate only needs a shape
/// that exercises every phase, so it keeps the canvas small and the decode
/// short. Override the three sizes from the command line when a probe needs
/// the release shape.
const WIDTH: i32 = 512;
const HEIGHT: i32 = 512;
const FRAMES: i32 = 22;
const STEPS: usize = 4;
const SEED: u64 = 42;

fn main() -> ExitCode {
    let mut width = WIDTH;
    let mut height = HEIGHT;
    let mut frames = FRAMES;
    let mut steps = STEPS;
    let mut vae_attention = Some(VaeAttention::Tensor);
    let mut vae_gemm = Some(VaeGemm::Reduced);
    let mut draft: Option<Draft> = None;

    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = |name: &str| -> Option<i64> {
            argv.next().and_then(|raw| match raw.trim().parse() {
                Ok(parsed) => Some(parsed),
                Err(_) => {
                    eprintln!("{name} needs an integer");
                    None
                }
            })
        };
        match flag.as_str() {
            "--width" => width = value("--width").unwrap_or(width as i64) as i32,
            "--height" => height = value("--height").unwrap_or(height as i64) as i32,
            "--frames" => frames = value("--frames").unwrap_or(frames as i64) as i32,
            "--steps" => steps = value("--steps").unwrap_or(steps as i64) as usize,
            // The decoder's arms. The gate keeps the campaign's fast arm so a
            // repeat of the residency numbers stays comparable; `env` leaves
            // them to `H3_VAE_ATTENTION`/`H3_VAE_GEMM`, which is the only way
            // to run the `bf16i` arm this host measured at 48 s.
            "--vae-arms" => match argv.next().as_deref() {
                Some("env") => {
                    vae_attention = None;
                    vae_gemm = None;
                }
                other => {
                    eprintln!("--vae-arms takes env, not {other:?}");
                    return ExitCode::from(2);
                }
            },
            "--vae-attention" => match argv.next().as_deref() {
                Some("scalar") => vae_attention = Some(VaeAttention::Scalar),
                Some("tensor") => vae_attention = Some(VaeAttention::Tensor),
                other => {
                    eprintln!("--vae-attention takes scalar or tensor, not {other:?}");
                    return ExitCode::from(2);
                }
            },
            "--vae-gemm" => match argv.next().as_deref() {
                Some("fp32") => vae_gemm = Some(VaeGemm::Fp32),
                Some("bf16") => vae_gemm = Some(VaeGemm::Reduced),
                other => {
                    eprintln!("--vae-gemm takes fp32 or bf16, not {other:?}");
                    return ExitCode::from(2);
                }
            },
            // The two-stage route (`harness/workflow.py:50`). `--draft` alone
            // is the reference's 0.7 / 0.72; the sigma is the only knob the
            // route's arithmetic has not already fixed.
            "--draft" => draft = Some(Draft::RELEASED),
            "--refine-sigma" => match argv.next().as_deref().map(str::parse::<f32>) {
                Some(Ok(sigma)) => {
                    draft = Some(Draft {
                        scale: draft.unwrap_or(Draft::RELEASED).scale,
                        sigma,
                    });
                }
                _ => {
                    eprintln!("--refine-sigma takes a number such as 0.72");
                    return ExitCode::from(2);
                }
            },
            // The upscaler's checkpoint directory, as the engine wants it:
            // through the environment, never as a path in the engine.
            "--upscaler" => match argv.next() {
                Some(dir) => std::env::set_var("H3_UPSCALER_DIR", dir),
                None => {
                    eprintln!("--upscaler needs a directory");
                    return ExitCode::from(2);
                }
            },
            "-h" | "--help" => {
                println!(
                    "usage: h3_engine_gate [--width N --height N --frames N --steps N] \
                     [--vae-arms env] [--vae-attention scalar|tensor] \
                     [--vae-gemm fp32|bf16] [--draft [--refine-sigma S]] [--upscaler DIR]"
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
    }

    // No default path: the checkpoint's location is the caller's setting, as it
    // is for every other H3 probe (`H3_MODEL`, or `H3_ARTIFACTS` plus the
    // checkpoint directory name in `H3_MODEL_NAME`).
    let model = model_dir();
    let Some(model) = model else {
        println!("SKIP H3_MODEL (or H3_ARTIFACTS + H3_MODEL_NAME) is not set");
        return ExitCode::from(3);
    };
    if !model.is_dir() {
        println!("SKIP missing model directory {}", model.display());
        return ExitCode::from(3);
    }

    let encoder = match std::env::var("H3_TEXT_NVFP4") {
        Ok(path) if !path.trim().is_empty() => Encoder::Quantized(PathBuf::from(path.trim())),
        _ => Encoder::Bf16,
    };

    println!(
        "ENGINE model={} canvas={}x{}x{} steps={} seed={} route={}",
        model.display(),
        width,
        height,
        frames,
        steps,
        SEED,
        match draft {
            Some(route) => format!("two-stage draft={} refine={}", route.scale, route.sigma),
            None => "direct".to_string(),
        }
    );

    let engine = match EngineHandle::spawn(&model) {
        Ok(engine) => engine,
        Err(error) => {
            println!("FAIL spawn: {}", error.message());
            return ExitCode::FAILURE;
        }
    };

    let dir = std::env::temp_dir().join(format!("h3-engine-gate-{}", std::process::id()));
    if let Err(error) = std::fs::create_dir_all(&dir) {
        println!("FAIL output directory: {error}");
        return ExitCode::FAILURE;
    }

    let mut reports = Vec::new();
    for turn in 1..=2 {
        let request = RenderRequest {
            prompt: "A red fox trots through fresh snow at dawn, crunching footsteps.".to_string(),
            width,
            height,
            frames,
            steps,
            reuse: 1,
            seed: SEED,
            draft,
            first_frame: None,
            last_frame: None,
            references: Vec::new(),
            attention: Attention::Tensor,
            vae_attention,
            vae_gemm,
            text_encoder: encoder.clone(),
            encode: Encode::default(),
            output: dir.join(format!("turn-{turn}.mp4")),
        };
        let started = std::time::Instant::now();
        match engine.render(request) {
            Ok(report) => {
                println!(
                    "REQ {turn} wall={:.3}s {}",
                    started.elapsed().as_secs_f64(),
                    report.summary()
                );
                // The two-stage split on its own line, so a log can be read
                // for the stage numbers without parsing the summary.
                println!(
                    "STAGES turn={turn} draft={:.3}s upscale={:.3}s refine={:.3}s \
                     load={:.3}s text_hit={} prepared_hit={} draft_audio={:016x} \
                     audio={:016x} frozen={}",
                    report.draft_seconds,
                    report.upscale_seconds,
                    report.refine_seconds,
                    report.load_seconds,
                    report.text_hit,
                    report.prepared_hit,
                    report.draft_audio_hash,
                    report.audio_hash,
                    report.audio_frozen
                );
                reports.push(report);
            }
            Err(error) => {
                println!("FAIL render {turn}: {}", error.message());
                return ExitCode::FAILURE;
            }
        }
    }

    let (first, second) = (&reports[0], &reports[1]);
    let identical = same_media(first, second);
    println!(
        "MEDIA identical={} first={} second={}",
        identical,
        media(first),
        media(second)
    );
    let residency = second.text_hit && second.prepared_hit;

    // The two-stage arm's own assertion: the audio the refine conditioned on is
    // the audio the decoder received, bit for bit, and the refine says so. This
    // is the reference's freeze (`harness/workflow.py:448`) and the property
    // `docs/kernels.md:296` records as the fix for the lip-sync failure mode.
    let frozen = reports
        .iter()
        .all(|report| report.audio_frozen && report.draft_audio_hash == report.audio_hash);

    if draft.is_some() {
        // Residency is not this arm's claim: the engine holds one prepared
        // graph and the two stages are two graphs, so a repeat reloads both.
        // What a repeat must still save is the prompt encode, and the stages
        // must not change the clip.
        if identical && frozen && second.text_hit {
            println!("RESULT PASS two-stage text_hit audio_frozen");
            return ExitCode::SUCCESS;
        }
        println!(
            "RESULT FAIL two-stage identical={} frozen={} text_hit={}",
            identical, frozen, second.text_hit
        );
        return ExitCode::FAILURE;
    }

    if residency && identical {
        println!("RESULT PASS residency");
        ExitCode::SUCCESS
    } else {
        println!(
            "RESULT FAIL text_hit={} prepared_hit={} identical={}",
            second.text_hit, second.prepared_hit, identical
        );
        ExitCode::FAILURE
    }
}

/// Where the checkpoint is, from the environment: `H3_MODEL` is the model
/// directory itself; otherwise `H3_ARTIFACTS` joined with `H3_MODEL_NAME`.
/// The gate assumes nothing about a host's layout.
fn model_dir() -> Option<PathBuf> {
    let set = |name: &str| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    };
    if let Some(model) = set("H3_MODEL") {
        return Some(PathBuf::from(model.trim()));
    }
    let artifacts = set("H3_ARTIFACTS")?;
    let name = set("H3_MODEL_NAME")?;
    Some(PathBuf::from(artifacts.trim()).join(name.trim()))
}

fn same_media(first: &RenderReport, second: &RenderReport) -> bool {
    first.frames == second.frames
        && first.video_hash == second.video_hash
        && first.audio_hash == second.audio_hash
        && first.frames_hash == second.frames_hash
}

fn media(report: &RenderReport) -> String {
    format!(
        "frames={} video={:016x} audio={:016x} frames={:016x}",
        report.frames, report.video_hash, report.audio_hash, report.frames_hash
    )
}
