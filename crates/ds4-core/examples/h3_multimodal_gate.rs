//! The multimodal gate: images plus a prompt, presented with the released
//! `<Picture n>: ` layout and encoded by the port's Qwen decoder.
//!
//! Usage: h3_multimodal_gate <FL2VA/text_encoder> <prompt> --image PATH
//!                            [--image PATH]... [--height H] [--width W]
//!                            [--expect HEX] [--tag-expect HEX]
//!        h3_multimodal_gate <FL2VA/text_encoder> <image> <prompt>
//!                            [--height H] [--width W]
//!
//! Each image is one `<Picture n>: ` span, in the order given
//! (`h3_multimodal.c:200-232`); the second spelling is the one-image form.
//!
//! Prints the grid and presentation size, then one PARITY line for the BF16
//! rows and one TAGS line, each with its FNV-1a hash and eight-word edges.
//! With `--expect` the process exits 3 when the row hash differs; with
//! `--tag-expect`, 4 when the tag hash differs.
//!
//! The image is read by `minimax_h3::media` at the requested shape, so both
//! sides see the same pixels when they are given the same file and shape. The
//! `<Picture 1>: ` prefix is tokenized by the released BPE; it is never
//! hardcoded.
//!
//! Goldens recorded on the Spark (gx10-e1d2, 2026-10-24) with the reference's
//! own `h3_multimodal_encode_fl2va_bf16` through
//! `tools/minimax_h3/multimodal_probe.c`, same image, prompt and shape. The
//! prompt is `A crocodile walks through the swamp at dusk.`; the image is
//! `tests/fixtures/h3/costello-crocodile-2panel.png`, read with `Fit::Stretch`,
//! one frame.
//!
//!   512x192 (grid 32x12, 96 merged rows, 114 rows total)
//!     PARITY fnv=6ced02fc6feb0fe6
//!     TAGS   fnv=7d8b4561c21600fb
//!   512x512 (grid 32x32, 256 merged rows, 274 rows total)
//!     PARITY fnv=9f8deeb453edfecc
//!     TAGS   fnv=0f10ec04569ce77b
//!
//! The port matched both hashes and both eight-word edges at each shape.
//! Wall time with a warm page cache: 5.245 s against the reference's 19.756 s
//! at 512x192 (the reference's run read the 62 GiB shard cold), and 7.028 s
//! against 8.570 s at 512x512.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::media::{read_image_f32, Fit};
use ds4_core::minimax_h3::multimodal;
use ds4_core::minimax_h3::text_forward::{self, Embedding};
use ds4_core::minimax_h3::tokenizer::Tokenizer;
use ds4_core::minimax_h3::vision::Grid;
use ds4_core::minimax_h3::vision_forward;

const DEFAULT_HEIGHT: u32 = 512;
const DEFAULT_WIDTH: u32 = 384;

/// The released tokenizer sits beside the shard it belongs to: the artifact
/// keeps `FL2VA/tokenizer/tokenizer.json` next to `FL2VA/text_encoder/`.
const TOKENIZER: &str = "../tokenizer/tokenizer.json";

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

    /* Two spellings, never mixed: `<prompt> --image ...`, or the legacy
     * `<image> <prompt>`. */
    let (prompt, images) = match (options.images.len(), positional.len()) {
        (0, 3) => (positional[2].clone(), vec![positional[1].clone()]),
        (0, _) => return Err(usage("expected <image> <prompt> or <prompt> --image")),
        (_, 2) => (positional[1].clone(), options.images.clone()),
        _ => return Err(usage("give <prompt> with --image, not both forms")),
    };

    let encoder = Path::new(&positional[0]);
    let height = options.height.unwrap_or(DEFAULT_HEIGHT);
    let width = options.width.unwrap_or(DEFAULT_WIDTH);
    let expect = options
        .expect
        .as_deref()
        .map(|text| parse_hash(text).ok_or_else(|| usage("--expect needs 16 hex digits")))
        .transpose()?;
    let tag_expect = options
        .tag_expect
        .as_deref()
        .map(|text| parse_hash(text).ok_or_else(|| usage("--tag-expect needs 16 hex digits")))
        .transpose()?;

    let grid = Grid::new(1, height as usize, width as usize)
        .map_err(|error| format!("grid: {}", error.token()))?;
    let tokenizer = Tokenizer::load(&encoder.join(TOKENIZER))
        .map_err(|error| format!("tokenizer: {}", error.token()))?;

    /* One tower run per image, in request order; the states stay alive until
     * the decoder has read every span. */
    let start = Instant::now();
    let mut outputs = Vec::with_capacity(images.len());

    for image in &images {
        let pixels = read_image_f32(Path::new(image), width, height, Fit::Stretch)
            .map_err(|error| format!("image {image}: {error}"))?;
        outputs.push(
            vision_forward::encode_dir(encoder, &pixels, 1, grid).map_err(|error| error.token())?,
        );
    }

    let vision: Vec<&vision_forward::VisionOutput> = outputs.iter().collect();
    let presentation =
        multimodal::fl2va(&tokenizer, &prompt, &vision).map_err(|error| error.token())?;
    let embedding = text_forward::encode_multimodal(
        encoder,
        presentation.ids(),
        presentation.spans(),
        presentation.positions(),
        presentation.tags(),
    )
    .map_err(|error| error.token())?;
    let seconds = start.elapsed().as_secs_f64();

    let (grid_h, grid_w) = grid.rows();
    let zeros = embedding
        .tags()
        .map(|tags| tags.iter().filter(|tag| **tag == 0).count())
        .unwrap_or(0);

    println!(
        "GRID {grid_h}x{grid_w} merged={} images={}",
        presentation.spans()[0].tokens,
        presentation.spans().len()
    );
    println!(
        "TOKENS {} spans={} zeros={zeros}",
        presentation.tokens(),
        presentation.spans().len()
    );
    println!(
        "{}",
        line("PARITY", &embedding, embedding.fnv(), embedding.values())
    );
    println!(
        "TAGS tokens={} fnv={:016x}",
        embedding.tokens(),
        embedding.tags_fnv().unwrap_or(0)
    );
    println!("SECONDS {seconds:.3}");

    if let Some(expect) = expect {
        if embedding.fnv() != expect {
            return Ok(ExitCode::from(3));
        }
    }
    if let Some(expect) = tag_expect {
        if embedding.tags_fnv() != Some(expect) {
            return Ok(ExitCode::from(4));
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// One hash line: shape, values, hash, and the two eight-word edges.
fn line(label: &str, embedding: &Embedding, hash: u64, payload: &[u8]) -> String {
    let total = payload.len() / 2;
    let edge = total.min(8);
    let last = total - edge;

    format!(
        "{label} tokens={} width={} values={} fnv={hash:016x} first={} last={}",
        embedding.tokens(),
        embedding.width(),
        total,
        hex_words(&bf16_words(payload, 0, edge)),
        hex_words(&bf16_words(payload, last, edge))
    )
}

/// Little-endian BF16 words of one span of a payload.
fn bf16_words(payload: &[u8], start: usize, count: usize) -> Vec<u16> {
    payload[start * 2..(start + count) * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

fn hex_words(values: &[u16]) -> String {
    values
        .iter()
        .map(|value| format!("{value:04x}"))
        .collect::<Vec<_>>()
        .join(".")
}

fn parse_hash(text: &str) -> Option<u64> {
    if text.len() != 16 {
        return None;
    }

    u64::from_str_radix(text, 16).ok()
}

/// The positional arguments, plus the flags the gate knows.
struct Options {
    height: Option<u32>,
    width: Option<u32>,
    images: Vec<String>,
    expect: Option<String>,
    tag_expect: Option<String>,
}

fn split(args: &[String]) -> Result<(Vec<String>, Options), String> {
    let mut positional = Vec::new();
    let mut options = Options {
        height: None,
        width: None,
        images: Vec::new(),
        expect: None,
        tag_expect: None,
    };
    let mut index = 0;

    while index < args.len() {
        let argument = &args[index];

        match argument.as_str() {
            "--image" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage("--image needs a path"))?;

                options.images.push(value.clone());
                index += 2;
            }
            "--height" | "--width" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage(&format!("{argument} needs a value")))?;
                let parsed = value
                    .parse::<u32>()
                    .map_err(|_| usage(&format!("{argument} needs a number")))?;

                if argument == "--height" {
                    options.height = Some(parsed);
                } else {
                    options.width = Some(parsed);
                }
                index += 2;
            }
            "--expect" | "--tag-expect" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage(&format!("{argument} needs a value")))?;

                if argument == "--expect" {
                    options.expect = Some(value.clone());
                } else {
                    options.tag_expect = Some(value.clone());
                }
                index += 2;
            }
            _ if argument.starts_with("--") => {
                return Err(usage(&format!("unknown option {argument}")));
            }
            _ => {
                positional.push(argument.clone());
                index += 1;
            }
        }
    }

    Ok((positional, options))
}

fn usage(detail: &str) -> String {
    format!(
        "{detail}\nusage: h3_multimodal_gate <FL2VA/text_encoder> <prompt> \
         --image PATH [--image PATH]... [--height H] [--width W] \
         [--expect HEX] [--tag-expect HEX]\n       \
         h3_multimodal_gate <FL2VA/text_encoder> <image> <prompt> \
         [--height H] [--width W]"
    )
}
