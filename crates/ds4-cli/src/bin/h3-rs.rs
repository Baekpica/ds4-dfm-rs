//! The h3 renderer: prompt to MP4, the way the reference CLI does it.
//!
//! This binary is the operator-facing end of the port and the renderer the
//! OpenAI Videos host spawns (`h3-server-rs --renderer`). Flags and defaults
//! are the reference's (`h3_cli.c:293-345`, `h3.h:131-135`,
//! `main.c:347-350`): `-d` model dir, `-p` prompt, `-o` output, `--width`
//! 864, `--height` 480, `--frames` 56, `--steps` 20, `--seed` 42, `--reuse` 1,
//! `--layers` 50.
//!
//! The pipeline is the reference's phase order (`h3.c:859-1700`): validate the
//! canvas, encode the prompt, build the sequence, denoise, decode audio and
//! video, convert and resize, mux.
//!
//! Two spellings differ on purpose and are recorded in the port plan:
//! `--ssd-streaming` selects the port's block ring (`LoadMode::Stream`), and
//! the flags the CUDA reference accepts as no-ops (`--use-int8-row-fc2`) are
//! not parsed at all rather than accepted and ignored.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::dit::{Dit, LoadMode};
use ds4_core::minimax_h3::host::{reuse_schedule, temporal, Rng, Schedule};
use ds4_core::minimax_h3::render::mux_mp4;
use ds4_core::minimax_h3::render::{check_frames, Canvas, RgbFrames};
use ds4_core::minimax_h3::sequence::{self, Spec};
use ds4_core::minimax_h3::text_forward;
use ds4_core::minimax_h3::tokenizer::{EmptyPrompt, Tokenizer};
use ds4_core::minimax_h3::{condition, media, vae_audio, vae_encoder, vae_video};

/// `h3.h:13-17`.
const DEFAULT_WIDTH: i32 = 864;
const DEFAULT_HEIGHT: i32 = 480;
const DEFAULT_FRAMES: i32 = 56;
const DEFAULT_STEPS: usize = 20;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_REUSE: usize = 1;
const DEFAULT_LAYERS: usize = 50;
const DEFAULT_OUTPUT: &str = "outputs/h3.mp4";

/// The component layout `h3.c:425-436` resolves under `-d`.
const FL2VA: &str = "FL2VA";

struct Options {
    model_dir: PathBuf,
    prompt: String,
    output: PathBuf,
    width: i32,
    height: i32,
    render_width: i32,
    render_height: i32,
    frames: i32,
    steps: usize,
    reuse: usize,
    layers: usize,
    seed: u64,
    /// `--first-frame` / `--last-frame` (`main.c:47-48`): the FL2VA anchors.
    first_frame: Option<PathBuf>,
    last_frame: Option<PathBuf>,
    reference_rope: bool,
    stream: bool,
    info: bool,
    hash: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("h3-rs: {detail}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let Some(options) = parse(std::env::args().skip(1).collect())? else {
        return Ok(ExitCode::SUCCESS);
    };

    let canvas = Canvas::new(
        options.width,
        options.height,
        options.render_width,
        options.render_height,
    )?;
    let frame_count = check_frames(options.frames)?;

    if options.info {
        return info(&options, &canvas, frame_count);
    }

    let component = |name: &str| options.model_dir.join(FL2VA).join(name);
    let tokenizer_path = component("tokenizer/tokenizer.json");
    let tokenizer = Tokenizer::load(&tokenizer_path)
        .map_err(|error| format!("tokenizer: {}", error.token()))?;

    let shape = temporal(frame_count);
    let (latent_w, latent_h) = canvas.latent();
    let anchors = read_anchors(&options, frame_count, latent_h, latent_w)?;
    if !anchors.keyframes.is_empty() {
        println!(
            "ANCHORS keyframes={} rows={}",
            anchors.keyframes.len(),
            anchors.rows.len() / condition::VIDEO_ROW_WIDTH
        );
    }

    let started = Instant::now();
    let text = match anchors.pixels.is_empty() {
        true => {
            let ids = tokenizer
                .encode(&options.prompt, EmptyPrompt::Pad)
                .map_err(|error| format!("tokenizer: {}", error.token()))?;
            text_forward::encode(&component("text_encoder"), &ids).map_err(|error| error.token())
        }
        // `h3.c:1345-1414`: an anchor is a vision span in the prompt, not only
        // a VAE row, so the text encoder runs after the tower.
        false => condition::anchor_embedding(
            &component("text_encoder"),
            &tokenizer,
            &options.prompt,
            &anchors.pixels,
        ),
    }?;
    println!(
        "TEXT tokens={} width={} fnv={:016x} seconds={:.3}",
        text.tokens(),
        text.width(),
        text.fnv(),
        started.elapsed().as_secs_f64()
    );

    let (keyframes, condition_rows) = (anchors.keyframes, anchors.rows);
    let spec = Spec {
        text_len: text.tokens(),
        latent_t: shape.video_t,
        latent_h,
        latent_w,
        audio_t: shape.audio_t,
        frame_count: shape.frame_count,
        keyframes: &keyframes,
    };
    let sequence = sequence::build(&spec).map_err(|error| format!("layout: {error:?}"))?;

    let schedule = Schedule::serving(options.steps)
        .ok_or_else(|| "--steps is outside 2..=1000".to_string())?;
    let mode = if options.stream {
        LoadMode::Stream
    } else {
        LoadMode::Resident
    };

    let started = Instant::now();
    let mut model = Dit::load_with(
        &component("transformer"),
        sequence,
        schedule,
        &text,
        canvas.rope_scale(options.reference_rope),
        mode,
    )
    .map_err(|error| error.to_string())?;
    println!(
        "LOAD seconds={:.3} layers={} latent={}x{}x{} frames={} audio_t={}",
        started.elapsed().as_secs_f64(),
        options.layers,
        shape.video_t,
        latent_h,
        latent_w,
        frame_count,
        shape.audio_t
    );

    if !condition_rows.is_empty() {
        model
            .set_conditions(&condition_rows, &[])
            .map_err(|error| error.to_string())?;
    }

    let mut video = vec![0.0f32; model.video_elements()];
    let mut audio = vec![0.0f32; model.audio_elements()];
    Rng::seed(options.seed).fill_normal(&mut video);
    Rng::seed(options.seed).fill_normal(&mut audio);

    let selected = reuse_schedule(options.steps, options.reuse)
        .ok_or_else(|| "--reuse is outside 1..=32".to_string())?;
    let evaluated = selected.iter().filter(|flag| **flag).count();

    let started = Instant::now();
    model
        .denoise(&mut video, &mut audio, options.reuse)
        .map_err(|error| error.to_string())?;
    println!(
        "DENOISE steps={} reuse={} selected={} seconds={:.3}",
        options.steps,
        options.reuse,
        evaluated,
        started.elapsed().as_secs_f64()
    );

    let video_hash = fnv_f32(&video);
    let audio_hash = fnv_f32(&audio);

    // Retire the DiT before the two VAEs are loaded: their weights are
    // allocated in the same unified pool the denoise just used.
    drop(model);

    let started = Instant::now();
    let waveform = vae_audio::decode(&component("audio_vae"), &audio, audio.len() / (32 * 2))
        .map_err(|error| error.to_string())?;
    println!(
        "AUDIO channels={} samples={} rate={} seconds={:.3}",
        waveform.channels,
        waveform.samples,
        waveform.sample_rate,
        started.elapsed().as_secs_f64()
    );

    let started = Instant::now();
    let decoded = vae_video::decode(
        &component("video_vae/source"),
        &video,
        shape.video_t as usize,
        latent_h as usize,
        latent_w as usize,
    )
    .map_err(|error| error.to_string())?;
    println!(
        "FRAMES frames={} height={} width={} seconds={:.3}",
        decoded.frames,
        decoded.height,
        decoded.width,
        started.elapsed().as_secs_f64()
    );

    let decoded_frames = small(decoded.frames, "decoded frames")?;
    let decoded_width = small(decoded.width, "decoded width")?;
    let decoded_height = small(decoded.height, "decoded height")?;
    let frames_hash = fnv_f32(&decoded.rgb);

    // `h3.c:1638-1660`: RGB24 at the decoded canvas, then the resize only when
    // the requested output canvas differs.
    let native = RgbFrames::from_f32(&decoded.rgb, decoded_frames, decoded_width, decoded_height)?;
    let mut frames = native.clone();

    if canvas.width() != native.width() || canvas.height() != native.height() {
        let started = Instant::now();
        frames = native.resized_to(canvas.width(), canvas.height())?;
        println!(
            "RESIZE {}x{} to {}x{} seconds={:.3}",
            native.width(),
            native.height(),
            frames.width(),
            frames.height(),
            started.elapsed().as_secs_f64()
        );
    }

    if options.hash {
        println!(
            "HASH video={:016x} audio={:016x} frames={:016x} rgb8={:016x} wrote={:016x}",
            video_hash,
            audio_hash,
            frames_hash,
            fnv_bytes(native.bytes()),
            fnv_bytes(frames.bytes())
        );
    }

    let program = std::env::var_os("H3_FFMPEG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffmpeg"));
    let started = Instant::now();
    mux_mp4(
        &program,
        &options.output,
        &frames,
        &waveform.pcm,
        small(waveform.samples, "audio samples")?,
        small(waveform.channels, "audio channels")?,
        small(waveform.sample_rate, "sample rate")?,
    )?;
    println!(
        "VIDEO path={} width={} height={} frames={} fps={} sample_rate={} seconds={:.3}",
        options.output.display(),
        frames.width(),
        frames.height(),
        frames.frames(),
        ds4_core::minimax_h3::render::FPS,
        waveform.sample_rate,
        started.elapsed().as_secs_f64()
    );

    Ok(ExitCode::SUCCESS)
}

fn info(options: &Options, canvas: &Canvas, frame_count: i32) -> Result<ExitCode, String> {
    let component = |name: &str| options.model_dir.join(FL2VA).join(name);
    let required = [
        component("tokenizer/tokenizer.json"),
        component("text_encoder"),
        component("transformer"),
        component("video_vae/source"),
        component("audio_vae"),
    ];

    for path in &required {
        if !path.exists() {
            return Err(format!("missing component {}", path.display()));
        }

        println!("COMPONENT path={}", path.display());
    }

    let (latent_w, latent_h) = canvas.latent();

    println!(
        "CANVAS {}x{} render={}x{} latent={}x{} rope_scale={} frames={} fps={}",
        canvas.width(),
        canvas.height(),
        canvas.render_width(),
        canvas.render_height(),
        latent_w,
        latent_h,
        canvas.rope_scale(options.reference_rope),
        frame_count,
        ds4_core::minimax_h3::render::FPS
    );

    Ok(ExitCode::SUCCESS)
}

/// The anchors of one request, read once: the pixels feed the vision tower and
/// the video VAE encoder, and the latent becomes the leading rows
/// (`h3.c:1231-1345`). Both anchors see the same jitter sequence because
/// `h3_augment_span` reseeds per span (`h3.c:315-322`).
struct Anchors {
    keyframes: Vec<i32>,
    rows: Vec<f32>,
    pixels: Vec<condition::AnchorPixels>,
}

fn read_anchors(
    options: &Options,
    frame_count: i32,
    latent_h: i32,
    latent_w: i32,
) -> Result<Anchors, String> {
    let anchors = [
        (0, media::Fit::Stretch, options.first_frame.as_deref()),
        (
            frame_count - 1,
            media::Fit::Cover,
            options.last_frame.as_deref(),
        ),
    ];
    let mut out = Anchors {
        keyframes: Vec::new(),
        rows: Vec::new(),
        pixels: Vec::new(),
    };

    for (frame, fit, path) in anchors {
        let Some(path) = path else { continue };
        let values = media::read_image_f32(path, options.width as u32, options.height as u32, fit)
            .map_err(|error| format!("anchor {}: {}", path.display(), error.detail()))?;
        let latent = vae_encoder::encode(
            &options.model_dir.join(FL2VA).join("video_vae/source"),
            &values,
            1,
            options.height as usize,
            options.width as usize,
            |_, _| {},
        )
        .map_err(|error| format!("anchor {}: {error}", path.display()))?;
        if latent.time != 1
            || latent.height != latent_h as usize
            || latent.width != latent_w as usize
        {
            return Err(format!(
                "anchor {}: VAE produced a {}x{}x{} latent, expected 1x{}x{}",
                path.display(),
                latent.time,
                latent.height,
                latent.width,
                latent_h,
                latent_w
            ));
        }
        out.keyframes.push(frame);
        out.rows.extend(
            condition::anchor_rows(&latent, options.seed)
                .map_err(|error| format!("anchor {}: {error}", path.display()))?,
        );
        out.pixels.push(condition::AnchorPixels {
            values,
            height: options.height as u32,
            width: options.width as u32,
        });
    }

    Ok(out)
}

fn parse(args: Vec<String>) -> Result<Option<Options>, String> {
    let mut options = Options {
        model_dir: PathBuf::new(),
        prompt: String::new(),
        output: PathBuf::from(DEFAULT_OUTPUT),
        width: DEFAULT_WIDTH,
        height: DEFAULT_HEIGHT,
        render_width: 0,
        render_height: 0,
        frames: DEFAULT_FRAMES,
        steps: DEFAULT_STEPS,
        reuse: DEFAULT_REUSE,
        layers: DEFAULT_LAYERS,
        seed: DEFAULT_SEED,
        first_frame: None,
        last_frame: None,
        reference_rope: false,
        stream: false,
        info: false,
        hash: false,
    };

    let mut argv = args.into_iter();

    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                print!("{}", usage());
                return Ok(None);
            }
            "-d" | "--model-dir" => options.model_dir = PathBuf::from(value(&mut argv, &flag)?),
            "-p" | "--prompt" => options.prompt = value(&mut argv, &flag)?,
            "-o" | "--output" => options.output = PathBuf::from(value(&mut argv, &flag)?),
            "--width" => options.width = integer(&mut argv, &flag)?,
            "--height" => options.height = integer(&mut argv, &flag)?,
            "--render-width" => options.render_width = integer(&mut argv, &flag)?,
            "--render-height" => options.render_height = integer(&mut argv, &flag)?,
            "--frames" => options.frames = integer(&mut argv, &flag)?,
            "--seconds" => {
                options.frames = seconds_to_frames(&value(&mut argv, &flag)?)?;
            }
            "--steps" => options.steps = unsigned(&mut argv, &flag)?,
            "--reuse" => options.reuse = unsigned(&mut argv, &flag)?,
            "--layers" => options.layers = unsigned(&mut argv, &flag)?,
            "--seed" => {
                options.seed = value(&mut argv, &flag)?
                    .parse()
                    .map_err(|_| "--seed is not an unsigned integer".to_string())?
            }
            "--first-frame" => options.first_frame = Some(PathBuf::from(value(&mut argv, &flag)?)),
            "--last-frame" => options.last_frame = Some(PathBuf::from(value(&mut argv, &flag)?)),
            "--use-reference-rope" => options.reference_rope = true,
            "--ssd-streaming" => options.stream = true,
            "--info" => options.info = true,
            "--hash" => options.hash = true,
            other => return Err(format!("unknown flag {other}; see --help")),
        }
    }

    if options.model_dir.as_os_str().is_empty() {
        return Err("-d MODEL_DIR is required".to_string());
    }

    if !options.info && options.prompt.is_empty() {
        return Err("-p PROMPT is required".to_string());
    }

    // The port runs the full stack; a truncated layer count is a diagnostic
    // the probe owns, and accepting it here would claim support it lacks.
    if options.layers != DEFAULT_LAYERS {
        return Err("--layers only supports the released 50-block stack".to_string());
    }

    Ok(Some(options))
}

fn value(argv: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    argv.next().ok_or_else(|| format!("{flag} needs a value"))
}

/// Decoded geometry is `usize` in the VAE APIs and the reference's `int` in
/// the canvas rules; refuse a value that cannot be one.
fn small<T: TryInto<i32>>(value: T, what: &str) -> Result<i32, String> {
    value
        .try_into()
        .map_err(|_| format!("{what} does not fit an i32"))
}

fn integer(argv: &mut impl Iterator<Item = String>, flag: &str) -> Result<i32, String> {
    value(argv, flag)?
        .parse()
        .map_err(|_| format!("{flag} is not an integer"))
}

fn unsigned(argv: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    value(argv, flag)?
        .parse()
        .map_err(|_| format!("{flag} is not an unsigned integer"))
}

/// The gate hash convention: FNV-1a over the little-endian bytes, the same
/// seed and prime the probe family uses (`text_forward::fnv1a`).
fn fnv_f32(values: &[f32]) -> u64 {
    let mut bytes = Vec::with_capacity(values.len() * 4);

    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    text_forward::fnv1a(&bytes)
}

fn fnv_bytes(values: &[u8]) -> u64 {
    text_forward::fnv1a(values)
}

/// `main.c:80-88`: seconds become frames at 24 fps, rounded.
fn seconds_to_frames(text: &str) -> Result<i32, String> {
    let seconds: f64 = text
        .parse()
        .map_err(|_| "--seconds is not a number".to_string())?;

    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("--seconds must be positive".to_string());
    }

    let frames = seconds * 24.0;

    if frames > f64::from(i32::MAX) {
        return Err("--seconds is too large".to_string());
    }

    Ok(frames.round() as i32)
}

fn usage() -> &'static str {
    "h3-rs - MiniMax-H3 FL2VA renderer

Usage: h3-rs -d MODEL_DIR -p PROMPT [options]

  -d, --model-dir DIR       checkpoint root holding FL2VA/ (required)
  -p, --prompt TEXT         prompt (required unless --info)
  -o, --output PATH         MP4 path (default outputs/h3.mp4)
  --width N                 output canvas width (default 864)
  --height N                output canvas height (default 480)
  --render-width N          internal render canvas width (default native)
  --render-height N         internal render canvas height (default native)
  --frames N                requested frames (default 56, aligned 5+17k)
  --seconds S               duration, converted at 24 fps
  --steps N                 denoise evaluations (default 20)
  --reuse N                 reuse window (default 1)
  --layers N                DiT blocks to run (default 50)
  --seed N                  seed, each modality (default 42)
  --first-frame PATH        anchor: the clip's first frame
  --last-frame PATH         anchor: the clip's last frame
  --use-reference-rope      rope scale 1.0 even at 256x256
  --ssd-streaming           stream the DiT blocks from the checkpoint
  --info                    resolve components and geometry, run nothing
  --hash                    print the gate hashes of the pipeline stages
  --help                    this text

Constraints (h3.c:488-520, h3.c:869-874): width and height are multiples of 32
of at least 32 with area at most 768*1344; the render canvas is optional, the
same aspect to the pixel and no larger than the output canvas; frames align to
5+17k and a render needs at least 22.

FFmpeg is taken from H3_FFMPEG when set, else from PATH.
"
}
