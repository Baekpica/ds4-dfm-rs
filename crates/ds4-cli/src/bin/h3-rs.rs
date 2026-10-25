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

use ds4_core::minimax_h3::dit::{Dit, DitError, LoadMode, LoadOptions, Reduction};
use ds4_core::minimax_h3::host::{reuse_schedule, temporal, Rng, Schedule};
use ds4_core::minimax_h3::render::mux_mp4;
use ds4_core::minimax_h3::render::{check_frames, Canvas, RgbFrames};
use ds4_core::minimax_h3::sequence::{self, Spec};
use ds4_core::minimax_h3::text_forward;
use ds4_core::minimax_h3::tokenizer::{EmptyPrompt, Tokenizer};
use ds4_core::minimax_h3::{
    condition, device, media, reference, render, terminal, vae_audio, vae_encoder, vae_video,
};

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
/// The ordered-reference partition (`h3.c:444-452`, selected at `:872-875`).
const REF2VA: &str = "Ref2VA";

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
    /// `--ref-image`, `--ref-video`, `--ref-silent-video`, `--ref-video-audio`
    /// and `--ref-audio` (`main.c:51-56`): ordered Ref2VA references.
    references: Vec<reference::Reference>,
    /// `--ref-image-size` (`main.c:52`, `:448-457`): `match` unless asked.
    reference_image_size: reference::ImageSize,
    reference_rope: bool,
    stream: bool,
    /// `--token-reduction` (`main.c:34`, `:306`, `:394`).
    token_reduction: bool,
    /// `--core-reuse` (`main.c:33`, `:305`, `:392`).
    core_reuse: u32,
    /// `--frames-dir` (`main.c:57`, `:339`, `:492`): the decoded frames.
    frames_dir: Option<PathBuf>,
    /// `--preview-dir` (`main.c:58`, `:340`, `:493`).
    preview_dir: Option<PathBuf>,
    /// `--show` (`main.c:59`, `:541-560`).
    show: bool,
    /// `--zoom` (`main.c:60`, `:342`, `:496`).
    zoom: u16,
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

    // `h3.c:597-600`: a request carries anchors or ordered references, never
    // both; refuse before any weight is read.
    if !options.references.is_empty()
        && (options.first_frame.is_some() || options.last_frame.is_some())
    {
        return Err("full references cannot be combined with frame anchors".to_string());
    }

    if options.info {
        return info(&options, &canvas, frame_count);
    }

    // `h3.c:872-875`: ordered references read the Ref2VA partition, and every
    // component comes from it (`h3.c:927-936`).
    let partition = if options.references.is_empty() {
        FL2VA
    } else {
        REF2VA
    };
    let component = |name: &str| options.model_dir.join(partition).join(name);
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

    // `h3.c:1027-1200`: each reference at its own canvas, in request order.
    let references = read_references(&options, &component, frame_count)?;
    if !references.layout.is_empty() {
        println!(
            "REFERENCES count={} video_rows={} audio_rows={} samples={} hash={:016x}",
            references.layout.len(),
            references.video_rows.len() / condition::VIDEO_ROW_WIDTH,
            references.audio_rows.len(),
            references.samples,
            references.hash
        );
    }

    let started = Instant::now();
    let text = if !references.pixels.is_empty() {
        // `h3.c:1345-1414`: a reference is a vision span in the prompt under
        // the Ref2VA presentation, so the encoder runs after the tower.
        condition::reference_embedding(
            &component("text_encoder"),
            &tokenizer,
            &options.prompt,
            &references.pixels,
        )
    } else if !anchors.pixels.is_empty() {
        condition::anchor_embedding(
            &component("text_encoder"),
            &tokenizer,
            &options.prompt,
            &anchors.pixels,
        )
    } else {
        let ids = tokenizer
            .encode(&options.prompt, EmptyPrompt::Pad)
            .map_err(|error| format!("tokenizer: {}", error.token()))?;
        text_forward::encode(&component("text_encoder"), &ids).map_err(|error| error.token())
    }?;
    println!(
        "TEXT tokens={} width={} fnv={:016x} seconds={:.3}",
        text.tokens(),
        text.width(),
        text.fnv(),
        started.elapsed().as_secs_f64()
    );

    let (keyframes, mut condition_rows) = (anchors.keyframes, anchors.rows);
    condition_rows.extend_from_slice(&references.video_rows);
    let spec = Spec {
        text_len: text.tokens(),
        latent_t: shape.video_t,
        latent_h,
        latent_w,
        audio_t: shape.audio_t,
        frame_count: shape.frame_count,
        keyframes: &keyframes,
        references: &references.layout,
    };
    let sequence = sequence::build(&spec).map_err(|error| format!("layout: {error:?}"))?;

    let schedule = Schedule::serving(options.steps)
        .ok_or_else(|| "--steps is outside 2..=1000".to_string())?;
    let mode = if options.stream {
        LoadMode::Stream
    } else {
        LoadMode::Resident
    };
    // `--token-reduction` sets the loader's flag; the window itself always
    // comes from `H3_TOKEN_REDUCTION_*` (`h3_dit.c:378-439`).
    let reduction = if options.token_reduction {
        Reduction::Enabled
    } else {
        Reduction::Environment
    };
    let load_options = LoadOptions {
        mode,
        layers: options.layers as u32,
        reduction,
        core_reuse: options.core_reuse,
    };

    let started = Instant::now();
    let mut model = Dit::load_with(
        &component("transformer"),
        sequence,
        schedule,
        &text,
        canvas.rope_scale(options.reference_rope),
        load_options,
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

    if !condition_rows.is_empty() || !references.audio_rows.is_empty() {
        model
            .set_conditions(&condition_rows, &references.audio_rows)
            .map_err(|error| error.to_string())?;
    }

    let mut video = vec![0.0f32; model.video_elements()];
    let mut audio = vec![0.0f32; model.audio_elements()];
    Rng::seed(options.seed).fill_normal(&mut video);
    Rng::seed(options.seed).fill_normal(&mut audio);

    let selected = reuse_schedule(options.steps, options.reuse)
        .ok_or_else(|| "--reuse is outside 1..=32".to_string())?;
    let evaluated = selected.iter().filter(|flag| **flag).count();

    // `main.c:540-560`: the preview decoder is loaded once, beside the DiT,
    // and `--show` warns and falls back to no display when the terminal has no
    // graphics protocol. Neither path touches the latents that are decoded
    // into the file.
    let mut preview = Preview::prepare(
        &options,
        shape.video_t as usize,
        latent_h,
        latent_w,
        component("video_vae/source"),
    )?;

    let started = Instant::now();
    match preview.as_mut() {
        Some(sink) => {
            let mut hook = |step: usize, latent: &[f32]| sink.deliver(step, latent);
            model
                .denoise_with(&mut video, &mut audio, options.reuse, Some(&mut hook))
                .map_err(|error| error.to_string())?;
        }
        None => model
            .denoise(&mut video, &mut audio, options.reuse)
            .map_err(|error| error.to_string())?,
    }
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

    // `main.c:214-245`: `--frames-dir` writes every final frame as a PPM,
    // after the resize and before the mux.
    if let Some(dir) = &options.frames_dir {
        write_frames(dir, &frames)?;
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

/// The denoise preview (`main.c:196-260`, `:540-575`): one decoded frame after
/// every step, written to `--preview-dir` and/or displayed by the terminal.
struct Preview {
    decoder: vae_video::PreviewDecoder,
    dir: Option<PathBuf>,
    protocol: terminal::Protocol,
    width: i32,
    height: i32,
    latent_t: usize,
    steps: usize,
}

impl Preview {
    fn prepare(
        options: &Options,
        latent_t: usize,
        latent_h: i32,
        latent_w: i32,
        vae: PathBuf,
    ) -> Result<Option<Preview>, String> {
        let protocol = match options.show {
            true => terminal::detect(),
            false => terminal::Protocol::None,
        };
        if options.show && protocol == terminal::Protocol::None {
            eprintln!("h3-rs: warning: --show needs Kitty, Ghostty, iTerm2, WezTerm, or Konsole");
        }
        if options.preview_dir.is_none() && protocol == terminal::Protocol::None {
            return Ok(None);
        }
        if !terminal::set_zoom(options.zoom) {
            return Err("--zoom must be at least 1".to_string());
        }
        if let Some(dir) = &options.preview_dir {
            std::fs::create_dir_all(dir).map_err(|error| {
                format!("cannot create preview directory {}: {error}", dir.display())
            })?;
        }

        let decoder = vae_video::PreviewDecoder::load(&vae, latent_h as usize, latent_w as usize)
            .map_err(|error| error.to_string())?;

        Ok(Some(Preview {
            decoder,
            dir: options.preview_dir.clone(),
            protocol,
            width: options.width,
            height: options.height,
            latent_t,
            steps: options.steps,
        }))
    }

    /// `h3_deliver_denoise_preview` (`h3.c:755-830`): decode the middle
    /// chunk's frame, convert and resize to the output canvas, then write it
    /// and/or display it. A display failure only disables the terminal
    /// (`main.c:257-260`); a file failure ends the render.
    fn deliver(&mut self, step: usize, latent: &[f32]) -> Result<(), DitError> {
        let (frames, index) = self
            .decoder
            .decode(latent, self.latent_t)
            .map_err(|error| DitError::Gpu(error.to_string()))?;
        let mut rgb = render::rgb_f32_to_u8(&frames.rgb);
        if frames.width != self.width as usize || frames.height != self.height as usize {
            rgb = render::resize_rgb24(
                &rgb,
                1,
                frames.width as i32,
                frames.height as i32,
                self.width,
                self.height,
            )
            .map_err(DitError::Gpu)?;
        }

        if let Some(dir) = &self.dir {
            write_ppm(dir, step, self.width as u32, self.height as u32, &rgb)
                .map_err(DitError::Gpu)?;
        }

        if self.protocol == terminal::Protocol::None {
            return Ok(());
        }
        match terminal::display_rgb24(self.protocol, &rgb, self.width as u32, self.height as u32) {
            Ok(()) => eprintln!(
                "h3-rs: denoise preview {}/{}, video frame {}/{} via {}",
                step + 1,
                self.steps,
                index + 1,
                (self.latent_t - 2) / 5 * 17 + 5,
                terminal::protocol_name(self.protocol)
            ),
            Err(error) => {
                eprintln!("h3-rs: terminal display disabled: {error}");
                self.protocol = terminal::Protocol::None;
            }
        }

        Ok(())
    }
}

/// `cli_write_ppm` (`main.c:196-213`): a scratch name and a rename, so a
/// reader never sees a partial preview.
fn write_ppm(
    dir: &std::path::Path,
    step: usize,
    width: u32,
    height: u32,
    rgb: &[u8],
) -> Result<(), String> {
    use std::io::Write;

    let staging = dir.join(".step.ppm");
    let path = dir.join(format!("step-{step:04}.ppm"));
    let mut file = std::fs::File::create(&staging)
        .map_err(|error| format!("cannot write preview {}: {error}", staging.display()))?;
    write!(file, "P6\n{width} {height}\n255\n")
        .and_then(|_| file.write_all(rgb))
        .map_err(|error| format!("cannot write preview {}: {error}", staging.display()))?;
    drop(file);
    std::fs::rename(&staging, &path)
        .map_err(|error| format!("cannot rename preview {}: {error}", path.display()))
}

/// `cli_frame`'s frame branch (`main.c:214-245`): `--frames-dir` gets one
/// `frame-%04d.ppm` per generated frame, written straight rather than through
/// a scratch name, as the reference does for final frames.
fn write_frames(dir: &std::path::Path, frames: &RgbFrames) -> Result<(), String> {
    use std::io::Write;

    std::fs::create_dir_all(dir)
        .map_err(|error| format!("cannot create frame directory {}: {error}", dir.display()))?;
    let (width, height) = (frames.width() as u32, frames.height() as u32);
    let row_bytes = frames.width() as usize * 3;
    let length = frames.frames() as usize;

    for index in 0..length {
        let path = dir.join(format!("frame-{index:04}.ppm"));
        let start = index * height as usize * row_bytes;
        let rgb = &frames.bytes()[start..start + height as usize * row_bytes];
        let mut file = std::fs::File::create(&path)
            .map_err(|error| format!("cannot write frame {index} to {}: {error}", dir.display()))?;
        write!(file, "P6\n{width} {height}\n255\n")
            .and_then(|_| file.write_all(rgb))
            .map_err(|_| format!("incomplete frame {index} in {}", dir.display()))?;
    }

    Ok(())
}

fn info(options: &Options, canvas: &Canvas, frame_count: i32) -> Result<ExitCode, String> {
    // `h3_load_dir` probes while opening the context and fails on an error
    // (`h3.c:456-463`); the port has no context here, so the probe runs once
    // and reports the same five CUDA fields `main.c:128-139` prints.
    let device = device::probe().map_err(|error| format!("device: {error}"))?;
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    println!(
        "DEVICE name=\"{}\" architecture=\"{}\" physical_gib={:.1} working_set_gib={:.1} max_buffer_gib={:.1} unified={}",
        device.name,
        device.architecture,
        gib(device.physical_memory),
        gib(device.recommended_working_set),
        gib(device.max_buffer_length),
        if device.unified_memory { "yes" } else { "no" }
    );

    // `h3.c:872-875` applies to `--info` too: the component probe names the
    // partition the run below would read.
    let partition = if options.references.is_empty() {
        FL2VA
    } else {
        REF2VA
    };
    let component = |name: &str| options.model_dir.join(partition).join(name);
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

/// Prepare the ordered references the way the resident engine does: one shared
/// port (`h3.c:1027-1200`), so the CLI and `--model-dir` cannot drift.
fn read_references(
    options: &Options,
    component: &impl Fn(&str) -> PathBuf,
    frame_count: i32,
) -> Result<reference::Set, String> {
    let (video_vae, audio_vae) = (component("video_vae/source"), component("audio_vae"));
    let paths = reference::Paths {
        video_vae: &video_vae,
        audio_vae: &audio_vae,
    };

    reference::prepare_set(
        &paths,
        &options.references,
        options.reference_image_size,
        (options.width as u32, options.height as u32),
        frame_count as usize,
        options.seed,
    )
    .map_err(|error| error.message().to_string())
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
        references: Vec::new(),
        reference_image_size: reference::ImageSize::Match,
        reference_rope: false,
        stream: false,
        token_reduction: false,
        core_reuse: 1,
        frames_dir: None,
        preview_dir: None,
        show: false,
        zoom: 2,
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
            // `main.c:429-490`: each flag appends one ordered reference; the
            // video forms differ only in whether the embedded track is read.
            "--ref-image" => options.references.push(reference::Reference {
                kind: reference::Kind::Image,
                path: PathBuf::from(value(&mut argv, &flag)?),
                audio_path: None,
                include_embedded_audio: false,
            }),
            "--ref-image-size" => {
                let text = value(&mut argv, &flag)?;
                options.reference_image_size = match text.as_str() {
                    "match" => reference::ImageSize::Match,
                    "max" => reference::ImageSize::Max,
                    other => {
                        return Err(format!(
                            "--ref-image-size must be match or max, not {other}"
                        ))
                    }
                };
            }
            "--ref-video" => options.references.push(reference::Reference {
                kind: reference::Kind::Video,
                path: PathBuf::from(value(&mut argv, &flag)?),
                audio_path: None,
                include_embedded_audio: true,
            }),
            "--ref-silent-video" => options.references.push(reference::Reference {
                kind: reference::Kind::Video,
                path: PathBuf::from(value(&mut argv, &flag)?),
                audio_path: None,
                include_embedded_audio: false,
            }),
            "--ref-video-audio" => {
                let path = PathBuf::from(value(&mut argv, &flag)?);
                let audio = PathBuf::from(value(&mut argv, &flag)?);
                options.references.push(reference::Reference {
                    kind: reference::Kind::VideoAudio,
                    path,
                    audio_path: Some(audio),
                    include_embedded_audio: false,
                });
            }
            "--ref-audio" => options.references.push(reference::Reference {
                kind: reference::Kind::Audio,
                path: PathBuf::from(value(&mut argv, &flag)?),
                audio_path: None,
                include_embedded_audio: false,
            }),
            "--use-reference-rope" => options.reference_rope = true,
            "--ssd-streaming" => options.stream = true,
            "--token-reduction" => options.token_reduction = true,
            "--frames-dir" => options.frames_dir = Some(PathBuf::from(value(&mut argv, &flag)?)),
            "--preview-dir" => options.preview_dir = Some(PathBuf::from(value(&mut argv, &flag)?)),
            "--show" => options.show = true,
            "--zoom" => {
                options.zoom = unsigned(&mut argv, &flag)?
                    .try_into()
                    .map_err(|_| "--zoom is larger than 65535".to_string())?
            }
            "--core-reuse" => {
                options.core_reuse = unsigned(&mut argv, &flag)?
                    .try_into()
                    .map_err(|_| "--core-reuse is larger than 6".to_string())?
            }
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
    "h3-rs - MiniMax-H3 renderer

Usage: h3-rs -d MODEL_DIR -p PROMPT [options]

  -d, --model-dir DIR       checkpoint root holding FL2VA/ and Ref2VA/ (required)
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
  --layers N                DiT blocks: 50 exact, 45 fast, 40 aggressive
  --seed N                  seed, each modality (default 42)
  --first-frame PATH        anchor: the clip's first frame
  --last-frame PATH         anchor: the clip's last frame
  --ref-image PATH          append an ordered Ref2VA image
  --ref-image-size S        image sizing: match (default) or max
  --ref-video PATH          append a video, including embedded audio
  --ref-silent-video PATH   append a video without its audio
  --ref-video-audio VIDEO AUDIO  append a video and a separate soundtrack
  --ref-audio PATH          append an ordered standalone audio clip
  --use-reference-rope      rope scale 1.0 even at 256x256
  --ssd-streaming           stream the DiT blocks from the checkpoint
  --token-reduction         pair video tokens in middle DiT blocks
                            (window from H3_TOKEN_REDUCTION_* env)
  --core-reuse N            core refresh: 1 exact, 4 fast, 6 aggressive
  --frames-dir PATH         write one PPM per generated frame
  --preview-dir PATH        write one PPM preview per denoising step
  --show                    display a preview frame per step (Kitty/iTerm2)
  --zoom N                  preview zoom for the terminal (default 2)
  --info                    resolve the device, components and geometry, run nothing
  --hash                    print the gate hashes of the pipeline stages
  --help                    this text

Constraints (h3.c:488-520, h3.c:869-874): width and height are multiples of 32
of at least 32 with area at most 768*1344; the render canvas is optional, the
same aspect to the pixel and no larger than the output canvas; frames align to
5+17k and a render needs at least 22.

FFmpeg is taken from H3_FFMPEG when set, else from PATH.
"
}
