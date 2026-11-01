//! The latent path's single command: a stage-1 draft, the x2 upscaler and the
//! H3-to-LTX adapter, three stage-2 steps, the LTX VAE, and the mux.
//!
//! Inputs are the two files `h3-rs` writes beside a stage-1 render:
//! `--save-latent` (raw BF16 `[channel][time][height][width]`, 24 channels,
//! before any VAE touches it) and `--save-audio` (raw little-endian F32,
//! interleaved). The route is the reference's (`runtime/stage2.py:58-76`,
//! `refiner_encoder_ablation_single_gpu.py:691-721`):
//!
//! ```text
//! draft latent -> upscaler (x2) -> adapter (align, 22 blocks, 2x shuffle)
//!   -> crop to 16 latent frames -> 3 Euler steps over SIGMAS -> LTX VAE
//!   -> frames -> mux with the stage-1 soundtrack
//! ```
//!
//! Every stage is the ported module the gates exercise; this binary only
//! composes them and checks each handoff's shape, so a mismatch fails loudly
//! instead of decoding something the reference never produced.
//!
//! Usage:
//!   h3-sol-rs --latent DRAFT --audio PCM --rate N --channels N
//!     --upscaler FILE --adapter DIR --refiner FILE --vae FILE --context FILE
//!     [--lora FILE] [--frames 124] [--draft-width 672] [--draft-height 384]
//!     [--seed N] [--sol] [-o OUT.mp4]

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::ltx_adapter::{self, Adapter, Arm, Upscaler};
use ds4_core::minimax_h3::ltx_audio_vae::{self, Encoder as AudioEncoder};
use ds4_core::minimax_h3::ltx_refiner::{
    Geometry as RefinerGeometry, Lora, AUDIO_CONTEXT_DIM, LAYERS,
};
use ds4_core::minimax_h3::ltx_stage2::Stage2;
use ds4_core::minimax_h3::ltx_vae::{pixel_shape, release_latent, LtxVae, RELEASE_LATENT};
use ds4_core::minimax_h3::prompt_cache::{self, FIXED_PROMPT};
use ds4_core::minimax_h3::render::{mux_mp4, RgbFrames};
use ds4_core::minimax_h3::vae_video;
use ds4_core::minimax_h3::torch_file::TorchDtype;
use ds4_core::minimax_h3::vision::bf16_to_f32;
use ds4_core::minimax_h3::h3_dit::{Dtype, Gpu, Tensor};

/// The release's stage-1 draft: 672x384 at 124 frames (`stage2.py`'s input).
const DEFAULT_FRAMES: usize = 124;
const DEFAULT_DRAFT_HEIGHT: usize = 384;
const DEFAULT_DRAFT_WIDTH: usize = 672;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_OUTPUT: &str = "outputs/h3-sol.mp4";
/// `configs/default.json:53`: the distilled refiner LoRA is applied at 0.8.
const DEFAULT_LORA_STRENGTH: f32 = 0.8;

/// The refiner's text context: the cached prompt's own shape
/// (`prompt_cache.py`; `CONTEXT_ROWS` in `ltx_refiner`).
const CONTEXT_ROWS: usize = 1024;
const CONTEXT_WIDTH: usize = 4096;

struct Options {
    latent: PathBuf,
    audio: PathBuf,
    audio_vae: Option<PathBuf>,
    no_audio_state: bool,
    rate: i32,
    channels: i32,
    upscaler: PathBuf,
    adapter: PathBuf,
    refiner: PathBuf,
    vae: PathBuf,
    context: PathBuf,
    lora: Option<PathBuf>,
    lora_strength: f32,
    frames: usize,
    draft_width: usize,
    draft_height: usize,
    seed: u64,
    sol: bool,
    /// `--hash`: the decoded frames' FNV, the same value the resident route
    /// prints. It is how the two lifetimes are compared: same request, same
    /// arm, same frames.
    hash: bool,
    /// `--sigmas A,B,...`: the refiner's own schedule, release default. A
    /// lower first value is the noise scale, so it keeps more of the draft.
    sigmas: Vec<f32>,
    /// Decode the upscaler and adapter output without running the refiner.
    transfer_only: bool,
    /// `--h3-vae DIR --h3-video draft|upscaled`: decode that H3 latent with the
    /// H3 video VAE and stop. The isolation arm for the upscaler: the same
    /// decoder, the same output canvas, only the latent differs.
    h3_vae: Option<PathBuf>,
    h3_video: Option<String>,
    /// `--save-upscaled PATH`: the upscaler's own BF16 `[c][t][h][w]` latent.
    upscaled_out: Option<PathBuf>,
    output: PathBuf,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(detail) => {
            eprintln!("h3-sol-rs: {detail}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let Some(options) = parse(std::env::args().skip(1).collect())? else {
        return Ok(ExitCode::SUCCESS);
    };

    // The draft's geometry is the reference's: the upscaler doubles the
    // canvas, the adapter takes the doubled one, and the crop to 16 latent
    // frames is what the refiner was built for.
    let geometry = ltx_adapter::geometry(options.frames, options.draft_height, options.draft_width)
        .map_err(|error| error.to_string())?;
    let [channels, time, height, width] = geometry.upscaler_input;
    println!(
        "DRAFT frames={} canvas={}x{} latent={}x{}x{} upscaled={:?} ltx={:?}",
        options.frames,
        geometry.draft_width,
        geometry.draft_height,
        channels,
        time,
        height * width,
        geometry.h3,
        geometry.ltx
    );

    let started = Instant::now();
    let latent = std::fs::read(&options.latent)
        .map_err(|error| format!("{}: {error}", options.latent.display()))?;
    let expected = geometry.upscaler_input_elements() * 2;

    if latent.len() != expected {
        return Err(format!(
            "{} holds {} bytes, a {}x{}x{} draft needs {expected}",
            options.latent.display(),
            latent.len(),
            channels,
            time,
            height * width
        ));
    }

    let gpu = Gpu::create().map_err(|detail| format!("device: {detail}"))?;
    let draft = upload_bf16(&gpu, &latent)?;
    println!("LOAD draft_bytes={} seconds={:.3}", latent.len(), started.elapsed().as_secs_f64());

    // The latent path's two models, in the reference's order. Both are freed
    // before the refiner allocates: this host shares one memory pool.
    let started = Instant::now();
    let upscaler = Upscaler::load(&options.upscaler, Arm::Bf16, &geometry)
        .map_err(|error| error.to_string())?;
    let upscaled = upscaler.forward(&draft).map_err(|error| error.to_string())?;

    // The upscaler's own latent, before the adapter or any refiner touches it.
    if let Some(path) = &options.upscaled_out {
        write_latent(&gpu, path, &upscaled)?;
    }

    // The isolation arm stops here: one decoder, one canvas, two latents.
    if let Some(which) = options.h3_video.clone() {
        return decode_h3(&gpu, &options, &which, &draft, &upscaled, &geometry);
    }
    let adapter = Adapter::load(&options.adapter, Arm::Bf16, &geometry)
        .map_err(|error| error.to_string())?;
    let converted = adapter.convert(&upscaled).map_err(|error| error.to_string())?;
    gpu.sync().map_err(|detail| format!("sync: {detail}"))?;

    let mut bytes = vec![0u8; converted.bytes()];
    gpu.read(&converted, &mut bytes)
        .map_err(|detail| format!("latent readback: {detail}"))?;
    let cropped = crop_frames(&bytes, &geometry);
    println!(
        "LATENT-PATH upscaled={:?} converted={:?} refiner_input={} bytes seconds={:.3}",
        geometry.h3,
        geometry.ltx,
        cropped.len(),
        started.elapsed().as_secs_f64()
    );

    drop(upscaled);
    drop(converted);
    drop(adapter);
    drop(upscaler);
    drop(draft);

    let initial = upload_bf16(&gpu, &cropped)?;

    // The transfer-only arm: decode what the upscaler and adapter produced,
    // without the refiner. It answers whether a change comes from the transfer
    // or from the denoise, and it never loads the 42 GB refiner.
    if options.transfer_only {
        decode_and_mux(&gpu, &options, &initial)?;
        return Ok(ExitCode::SUCCESS);
    }

    // The cached stage-2 prompt: Gemma4 plus the connector, built offline.
    let cache = prompt_cache::load(&options.context, FIXED_PROMPT)
        .map_err(|error| format!("context: {error}"))?;

    // `cache_builder.py:129-150` stores each context as `[1, rows, inner]`;
    // the refiner reads it as `rows` x `inner`, so only the element count and
    // the dtype matter here.
    if cache.video.dtype != TorchDtype::BFloat16
        || cache.video.shape != [1, CONTEXT_ROWS as u64, CONTEXT_WIDTH as u64]
    {
        return Err(format!(
            "context video is {:?} {:?}, expected BF16 [1, {CONTEXT_ROWS}, {CONTEXT_WIDTH}]",
            cache.video.dtype, cache.video.shape
        ));
    }

    let context = upload_bf16(&gpu, &cache.video.bytes)?;
    let audio = if options.no_audio_state {
        println!("AUDIO state=off (ablation)");
        None
    } else {
        if cache.audio.dtype != TorchDtype::BFloat16
            || cache.audio.shape != [1, CONTEXT_ROWS as u64, AUDIO_CONTEXT_DIM as u64]
        {
            return Err(format!(
                "context audio is {:?} {:?}, expected BF16 [1, {CONTEXT_ROWS}, {AUDIO_CONTEXT_DIM}]",
                cache.audio.dtype, cache.audio.shape
            ));
        }

        let started = Instant::now();
        let pcm = read_pcm(&options.audio)?;
        let mel = ltx_audio_vae::log_mel(&pcm, options.channels as usize, options.rate as u32)
            .map_err(|error| error.to_string())?;
        let path = options.audio_vae.as_deref().ok_or("missing --audio-vae")?;
        let encoder = AudioEncoder::open(&gpu, path)
            .map_err(|error| error.to_string())?;
        let encoded = encoder.encode(&gpu, &mel).map_err(|error| error.to_string())?;
        let conformed = encoded.conformed(RefinerGeometry::release().audio_frames);
        let bytes = conformed.bf16_bytes();
        let samples: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|pair| bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
            .collect();
        let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
        let std = (samples.iter().map(|value| (value - mean).powi(2)).sum::<f32>()
            / samples.len() as f32)
            .sqrt();
        println!(
            "AUDIO state={}x{}x{} mean={mean:.5} std={std:.5} encode_seconds={:.3}",
            conformed.channels,
            conformed.frames,
            conformed.bins,
            started.elapsed().as_secs_f64()
        );
        Some((upload_bf16(&gpu, &bytes)?, upload_bf16(&gpu, &cache.audio.bytes)?))
    };
    let lora = match &options.lora {
        Some(path) => {
            println!(
                "LORA path={} strength={:.2}",
                path.display(),
                options.lora_strength
            );
            Some(Lora::open(path, options.lora_strength).map_err(|error| error.token())?)
        }
        None => None,
    };

    // The refiner's own geometry is the release's, not the draft's: 16 latent
    // frames of 24x42.
    let refiner_geometry = RefinerGeometry::release();
    let layers: Vec<usize> = (0..LAYERS).collect();
    let mut stage = Stage2::open(
        &gpu,
        &options.refiner,
        &options.refiner,
        refiner_geometry,
        &layers,
        lora.as_ref(),
    )
    .map_err(|error| error.token())?;

    if options.sol {
        stage.enable_sol(&gpu).map_err(|error| error.token())?;
    }

    if !options.sigmas.is_empty() {
        stage
            .set_schedule(&options.sigmas)
            .map_err(|error| error.token())?;
    }

    println!(
        "SIGMAS values={:?} steps={} release={}",
        stage.schedule(),
        stage.steps(),
        stage.is_release_schedule()
    );

    stage.seed(options.seed);

    let started = Instant::now();
    stage
        .noised_state(&gpu, &initial)
        .map_err(|error| error.token())?;
    println!("STAGE2 prepare={:.3}", started.elapsed().as_secs_f64());

    let started = Instant::now();

    for index in 0..stage.steps() {
        let step_started = Instant::now();
        stage
            .step(&gpu, index, &context, audio.as_ref().map(|(state, prompt)| (state, prompt)))
            .map_err(|error| error.token())?;
        println!(
            "STAGE2 step={index} seconds={:.3}",
            step_started.elapsed().as_secs_f64()
        );
    }

    let grid = stage.latent_grid(&gpu).map_err(|error| error.token())?;
    println!("STAGE2 denoise_seconds={:.3}", started.elapsed().as_secs_f64());

    decode_and_mux(&gpu, &options, grid)?;

    Ok(ExitCode::SUCCESS)
}

/// Decode a BF16 `[channel][frame][height][width]` latent with the LTX VAE and
/// mux the stage-1 PCM beside it. The refiner's grid and the transfer's own
/// output share this contract (`ltx_vae::decode`).
/// The upscaler's latent as raw BF16 `[channel][time][height][width]`, the
/// same layout `h3-rs --save-latent` writes and reads.
fn write_latent(gpu: &Gpu, path: &Path, tensor: &Tensor) -> Result<(), String> {
    let mut bytes = vec![0u8; tensor.bytes()];
    gpu.read(tensor, &mut bytes)
        .map_err(|detail| format!("latent readback: {detail}"))?;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("{}: {error}", parent.display()))?;
        }
    }

    std::fs::write(path, &bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    println!("UPSCALED path={} bytes={}", path.display(), bytes.len());

    Ok(())
}

/// Decode the draft's H3 latent or the upscaler's with the H3 video VAE and
/// mux it. Both arms run the same decoder to the same canvas, so the only
/// difference is the latent the upscaler did or did not produce.
fn decode_h3(
    gpu: &Gpu,
    options: &Options,
    which: &str,
    draft: &Tensor,
    upscaled: &Tensor,
    geometry: &ltx_adapter::Geometry,
) -> Result<ExitCode, String> {
    let directory = options
        .h3_vae
        .as_ref()
        .ok_or("--h3-video needs --h3-vae")?;
    let [channels, time, up_height, up_width] = geometry.h3;

    let (dims, values) = match which {
        "upscaled" => ([channels, time, up_height, up_width], read_bf16(gpu, upscaled)?),
        "draft" => (
            [channels, time, up_height / 2, up_width / 2],
            read_bf16(gpu, draft)?,
        ),
        other => return Err(format!("--h3-video takes draft or upscaled, not {other}")),
    };

    let started = Instant::now();
    let decoded = vae_video::decode(
        directory,
        &values,
        dims[1] as usize,
        dims[2] as usize,
        dims[3] as usize,
        // No flag states the decoder's arms on this route: the environment's
        // word (`H3_VAE_GEMM`, `H3_VAE_ATTENTION`) or the reference's default.
        vae_video::Arms::default(),
    )
    .map_err(|error| error.to_string())?;
    println!(
        "H3-VAE latent={which} frames={} height={} width={} seconds={:.3}",
        decoded.frames,
        decoded.height,
        decoded.width,
        started.elapsed().as_secs_f64()
    );
    if options.hash {
        println!("H3-VAE frames_fnv={:016x}", fnv_f32(&decoded.rgb));
    }

    let frames = RgbFrames::from_f32(
        &decoded.rgb,
        decoded.frames as i32,
        decoded.width as i32,
        decoded.height as i32,
    )?;
    let pcm = read_pcm(&options.audio)?;
    let samples = i32::try_from(pcm.len() / options.channels as usize)
        .map_err(|_| "audio is longer than the muxer takes".to_string())?;
    let program = std::env::var_os("H3_FFMPEG")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffmpeg"));
    mux_mp4(
        &program,
        &options.output,
        &frames,
        &pcm,
        samples,
        options.channels,
        options.rate,
    )?;
    println!("H3-VIDEO path={}", options.output.display());

    Ok(ExitCode::SUCCESS)
}

fn decode_and_mux(gpu: &Gpu, options: &Options, latent: &Tensor) -> Result<(), String> {
    let shape = release_latent();
    let started = Instant::now();
    let vae = LtxVae::open(gpu, &options.vae, shape).map_err(|error| error.to_string())?;
    let rgb = vae
        .decode(gpu, latent, shape)
        .map_err(|error| error.to_string())?;
    gpu.sync().map_err(|detail| format!("sync: {detail}"))?;
    println!("LTX-VAE decode_seconds={:.3}", started.elapsed().as_secs_f64());

    let pixel = pixel_shape(RELEASE_LATENT);
    let values = read_bf16(gpu, rgb)?;
    let frames = RgbFrames::from_f32(
        &values,
        pixel[0] as i32,
        pixel[2] as i32,
        pixel[1] as i32,
    )?;
    println!(
        "FRAMES frames={} height={} width={}",
        frames.frames(),
        frames.height(),
        frames.width()
    );
    if options.hash {
        println!("FRAMES frames_fnv={:016x}", fnv_f32(&values));
    }

    drop(vae);

    let pcm = read_pcm(&options.audio)?;
    let samples = i32::try_from(pcm.len() / options.channels as usize)
        .map_err(|_| "audio is longer than the muxer takes".to_string())?;
    let program = std::env::var_os("H3_FFMPEG")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ffmpeg"));
    let started = Instant::now();
    mux_mp4(
        &program,
        &options.output,
        &frames,
        &pcm,
        samples,
        options.channels,
        options.rate,
    )?;
    println!(
        "VIDEO path={} seconds={:.3}",
        options.output.display(),
        started.elapsed().as_secs_f64()
    );

    Ok(())
}

/// The reference's `latent[:, :, :16]` (`runtime/stage2.py:76`): the first 16
/// frames of each channel's own run, not a prefix of the buffer.
fn crop_frames(bytes: &[u8], geometry: &ltx_adapter::Geometry) -> Vec<u8> {
    let [channels, frames, height, width] = geometry.ltx;
    let keep = RELEASE_LATENT[0] as usize;
    let plane = height * width;
    let mut out = Vec::with_capacity(channels * keep * plane * 2);

    for channel in 0..channels {
        let start = channel * frames * plane * 2;
        out.extend_from_slice(&bytes[start..start + keep * plane * 2]);
    }

    out
}

fn upload_bf16(gpu: &Gpu, bytes: &[u8]) -> Result<Tensor, String> {
    let tensor = gpu
        .alloc(Dtype::Bf16, bytes.len() / 2)
        .map_err(|detail| format!("alloc: {detail}"))?;
    gpu.upload(&tensor, bytes)
        .map_err(|detail| format!("upload: {detail}"))?;

    Ok(tensor)
}

fn read_bf16(gpu: &Gpu, tensor: &Tensor) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0u8; tensor.bytes()];
    gpu.read(tensor, &mut bytes)
        .map_err(|detail| format!("readback: {detail}"))?;

    Ok(bytes
        .chunks_exact(2)
        .map(|pair| bf16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
        .collect())
}

/// The gate hash convention `h3-rs` and the engine use: FNV-1a over the
/// little-endian F32 bytes, so the two lifetimes compare the same number.
fn fnv_f32(values: &[f32]) -> u64 {
    use ds4_core::minimax_h3::text_forward::{FNV_OFFSET, FNV_PRIME};

    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .fold(FNV_OFFSET, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
        })
}

fn read_pcm(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;

    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not whole F32 samples", path.display()));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|sample| f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]))
        .collect())
}

/// A borrowed empty path, so the required-argument check can name a flag that
/// has no default.
static EMPTY: PathBuf = PathBuf::new();

fn parse(arguments: Vec<String>) -> Result<Option<Options>, String> {
    let mut options = Options {
        latent: PathBuf::new(),
        audio: PathBuf::new(),
        audio_vae: None,
        no_audio_state: false,
        rate: 32_000,
        channels: 2,
        upscaler: PathBuf::new(),
        adapter: PathBuf::new(),
        refiner: PathBuf::new(),
        vae: PathBuf::new(),
        context: PathBuf::new(),
        lora: None,
        lora_strength: DEFAULT_LORA_STRENGTH,
        frames: DEFAULT_FRAMES,
        draft_width: DEFAULT_DRAFT_WIDTH,
        draft_height: DEFAULT_DRAFT_HEIGHT,
        seed: DEFAULT_SEED,
        sol: false,
        hash: false,
        sigmas: Vec::new(),
        transfer_only: false,
        h3_vae: None,
        h3_video: None,
        upscaled_out: None,
        output: PathBuf::from(DEFAULT_OUTPUT),
    };
    let mut argv = arguments.into_iter();

    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                print!("{}", usage());
                return Ok(None);
            }
            "--latent" => options.latent = PathBuf::from(value(&mut argv, &flag)?),
            "--audio" => options.audio = PathBuf::from(value(&mut argv, &flag)?),
            "--audio-vae" => options.audio_vae = Some(PathBuf::from(value(&mut argv, &flag)?)),
            "--no-audio-state" => options.no_audio_state = true,
            "--rate" => options.rate = integer(&mut argv, &flag)?,
            "--channels" => options.channels = integer(&mut argv, &flag)?,
            "--upscaler" => options.upscaler = PathBuf::from(value(&mut argv, &flag)?),
            "--adapter" => options.adapter = PathBuf::from(value(&mut argv, &flag)?),
            "--refiner" => options.refiner = PathBuf::from(value(&mut argv, &flag)?),
            "--vae" => options.vae = PathBuf::from(value(&mut argv, &flag)?),
            "--context" => options.context = PathBuf::from(value(&mut argv, &flag)?),
            "--lora" => options.lora = Some(PathBuf::from(value(&mut argv, &flag)?)),
            "--lora-strength" => options.lora_strength = real(&mut argv, &flag)?,
            "--frames" => options.frames = unsigned(&mut argv, &flag)? as usize,
            "--draft-width" => options.draft_width = unsigned(&mut argv, &flag)? as usize,
            "--draft-height" => options.draft_height = unsigned(&mut argv, &flag)? as usize,
            "--seed" => options.seed = unsigned(&mut argv, &flag)?,
            "--sol" => options.sol = true,
            "--hash" => options.hash = true,
            "--h3-vae" => options.h3_vae = Some(PathBuf::from(value(&mut argv, &flag)?)),
            "--h3-video" => options.h3_video = Some(value(&mut argv, &flag)?),
            "--save-upscaled" => {
                options.upscaled_out = Some(PathBuf::from(value(&mut argv, &flag)?))
            }
            "--sigmas" => options.sigmas = csv(&mut argv, &flag)?,
            "--transfer-only" => options.transfer_only = true,
            "-o" | "--output" => options.output = PathBuf::from(value(&mut argv, &flag)?),
            other => return Err(format!("unknown flag {other}; see --help")),
        }
    }

    // The isolation arm stops after the upscaler, so it needs the H3 VAE and
    // nothing of the LTX side: no adapter, no refiner, no prompt context.
    let mut required = vec![
        ("--latent", &options.latent),
        ("--audio", &options.audio),
        ("--upscaler", &options.upscaler),
    ];

    if options.h3_video.is_some() {
        required.push(("--h3-vae", options.h3_vae.as_ref().unwrap_or(&EMPTY)));
    } else {
        required.push(("--adapter", &options.adapter));
        required.push(("--refiner", &options.refiner));
        required.push(("--vae", &options.vae));
        required.push(("--context", &options.context));
        if !options.no_audio_state && !options.transfer_only {
            required.push(("--audio-vae", options.audio_vae.as_ref().unwrap_or(&EMPTY)));
        }
    }

    let missing: Vec<&str> = required
        .iter()
        .filter(|(_, path)| path.as_os_str().is_empty())
        .map(|(flag, _)| *flag)
        .collect();

    if !missing.is_empty() {
        return Err(format!("missing {}", missing.join(", ")));
    }

    Ok(Some(options))
}

fn value(argv: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    argv.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn integer(argv: &mut impl Iterator<Item = String>, flag: &str) -> Result<i32, String> {
    value(argv, flag)?
        .parse()
        .map_err(|_| format!("{flag} is not an integer"))
}

fn unsigned(argv: &mut impl Iterator<Item = String>, flag: &str) -> Result<u64, String> {
    value(argv, flag)?
        .parse()
        .map_err(|_| format!("{flag} is not an unsigned integer"))
}

/// A finite, non-negative strength or scale.
fn real(argv: &mut impl Iterator<Item = String>, flag: &str) -> Result<f32, String> {
    let parsed: f32 = value(argv, flag)?
        .parse()
        .map_err(|_| format!("{flag} is not a number"))?;

    if !parsed.is_finite() || parsed < 0.0 {
        return Err(format!("{flag} must be a finite non-negative number"));
    }

    Ok(parsed)
}

/// A comma-separated refiner schedule, walked from the first value to zero.
fn csv(argv: &mut impl Iterator<Item = String>, flag: &str) -> Result<Vec<f32>, String> {
    let raw = value(argv, flag)?;
    let mut schedule = Vec::new();

    for piece in raw.split(',') {
        let parsed: f32 = piece
            .trim()
            .parse()
            .map_err(|_| format!("{flag} has a value that is not a number"))?;

        if !parsed.is_finite() || parsed < 0.0 {
            return Err(format!("{flag} must be finite and non-negative"));
        }

        schedule.push(parsed);
    }

    Ok(schedule)
}

fn usage() -> String {
    "\
Usage: h3-sol-rs --latent DRAFT --audio PCM --upscaler FILE --adapter DIR \
--refiner FILE --vae FILE --context FILE -o OUT.mp4 [options]

  --latent PATH             stage-1 latent: raw BF16 [channel][time][height][width]
  --audio PATH              soundtrack: raw little-endian F32, interleaved
  --audio-vae PATH          LTX audio VAE for the clean frozen audio state.
                            Opt-in: it costs about 11 s of denoise over the
                            video-only route
                            (docs/minimax-h3/av-cross-modulation-pair-2026-11-01.md)
  --no-audio-state          video-only ablation; omit the audio state and context
  --rate N                  soundtrack sample rate (default 32000)
  --channels N              soundtrack channels (default 2)
  --upscaler PATH           minimax_h3_latent_upscaler_3d checkpoint
  --adapter DIR             H3-to-LTX adapter directory
  --refiner PATH            LTX-2.5 transformer (edges shard is the same file)
  --vae PATH                LTX video VAE checkpoint
  --context PATH            cached stage-2 prompt (prompt_cache container)
  --lora PATH               distilled refiner LoRA, fused as weights are read
  --lora-strength N         LoRA strength (default 0.8, the release's value)
  --frames N                stage-1 pixel frames (default 124)
  --draft-width N           stage-1 canvas width (default 672)
  --draft-height N          stage-1 canvas height (default 384)
  --seed N                  stage-2 noise seed (default 42)
  --sol                     enable the release's Sol-Attn routing
  --hash                    print the decoded frames' FNV (how the resident
                            route and this one are compared)
  --save-upscaled PATH      write the upscaler's own H3 latent, [c][t][h][w] BF16
  --h3-vae DIR              H3 video VAE component, for the isolation arm
  --h3-video draft|upscaled decode that latent with the H3 VAE and stop
  --sigmas A,B,...          refiner schedule, release is 0.909375,0.725,0.421875,0
                            a lower first value keeps more of the draft
  --transfer-only           decode upscaler + adapter output, no refiner
  -o, --output PATH         MP4 path (default outputs/h3-sol.mp4)
  --help                    this text

The draft is the file h3-rs --save-latent writes, at the canvas h3-rs
--width 672 --height 384 --frames 124 rendered. The audio is h3-rs
--save-audio. FFmpeg is taken from H3_FFMPEG when set, else from PATH.
"
        .to_string()
}
