//! The vision-tower gate: the port's encoder against the reference's golden.
//!
//! Usage: h3_vision_gate <FL2VA/text_encoder> <image>
//!                        [--height H] [--width W] [--expect HEX]
//!
//! Prints the grid, then one PARITY line for the final merger and one
//! DEEPSTACK line per deepstack: shape, FNV-1a over the BF16 payload, and the
//! first and last eight BF16 words. With `--expect` the process exits 3 when
//! the merged hash differs.
//!
//! The image is read by `minimax_h3::media` (FFmpeg, Lanczos, channel-major)
//! at the requested shape, which must be a multiple of 32 on both edges.
//!
//! Goldens recorded on the Spark (gx10-e1d2, 2026-10-24) with the reference's
//! own `h3_vision_encode_bf16` through `tools/minimax_h3/vision_probe.c`, same
//! image and shape. The port matched all four hashes at each shape, and the
//! MERGED hash also at 512x224, 512x256, 512x320, 512x384 and 512x448:
//!
//!   512x192 (grid 32x12, 96 rows)
//!     PARITY   b583e7a6059acdc0
//!     DEEPSTACK 0 316017d0a0ccbc8e, 1 2d52638d8308c78d, 2 91ef1cab820cb4b3
//!   512x512 (grid 32x32, 256 rows)
//!     PARITY   defef5800918fc1b
//!     DEEPSTACK 0 b5ad09518b677e73, 1 92910bbc32884dbc, 2 9aae4a41ec5a194d
//!
//! The image is `tests/fixtures/h3/costello-crocodile-2panel.png`, read with
//! `Fit::Stretch`, one frame.

use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::media::{read_image_f32, Fit};
use ds4_core::minimax_h3::vision::{Grid, DEEPSTACKS};
use ds4_core::minimax_h3::vision_forward::{self, VisionOutput};

const DEFAULT_HEIGHT: u32 = 512;
const DEFAULT_WIDTH: u32 = 192;

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

    let encoder = std::path::Path::new(&positional[0]);
    let image = std::path::Path::new(&positional[1]);
    let height = options.height.unwrap_or(DEFAULT_HEIGHT);
    let width = options.width.unwrap_or(DEFAULT_WIDTH);
    let expect = match options.expect.as_deref() {
        Some(text) => {
            Some(parse_hash(text).ok_or_else(|| usage("--expect needs a hexadecimal hash"))?)
        }
        None => None,
    };
    let grid = Grid::new(1, height as usize, width as usize).map_err(|error| error.token())?;
    let pixels = read_image_f32(image, width, height, Fit::Stretch)
        .map_err(|error| format!("image: {error}"))?;

    let start = Instant::now();
    let output =
        vision_forward::encode_dir(encoder, &pixels, 1, grid).map_err(|error| error.token())?;
    let seconds = start.elapsed().as_secs_f64();

    let (grid_h, grid_w) = grid.rows();

    println!(
        "GRID {grid_h}x{grid_w} rows={} merged={}",
        grid.tokens(),
        output.tokens()
    );
    println!("{}", line("PARITY", &output, output.fnv(), output.merged()));

    for index in 0..DEEPSTACKS {
        println!(
            "{}",
            line(
                "DEEPSTACK",
                &output,
                output.deepstack_fnv(index),
                output.deepstack(index)
            )
        );
    }

    println!("SECONDS {seconds:.3}");

    if let Some(expect) = expect {
        if output.fnv() != expect {
            return Ok(ExitCode::from(3));
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// One hash line: shape, values, hash, and the two eight-word edges.
fn line(label: &str, output: &VisionOutput, hash: u64, payload: &[u8]) -> String {
    let total = payload.len() / 2;
    let edge = total.min(8);
    let last = total - edge;

    format!(
        "{label} tokens={} width={} values={} fnv={hash:016x} first={} last={}",
        output.tokens(),
        output.width(),
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

/// The two positional arguments, plus the flags the gate knows.
struct Options {
    height: Option<u32>,
    width: Option<u32>,
    expect: Option<String>,
}

fn split(args: &[String]) -> Result<(Vec<String>, Options), String> {
    let mut positional = Vec::new();
    let mut options = Options {
        height: None,
        width: None,
        expect: None,
    };
    let mut index = 0;

    while index < args.len() {
        let argument = &args[index];

        match argument.as_str() {
            "--height" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage("--height needs a value"))?;
                options.height = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| usage("--height needs a number"))?,
                );
                index += 2;
            }
            "--width" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage("--width needs a value"))?;
                options.width = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| usage("--width needs a number"))?,
                );
                index += 2;
            }
            "--expect" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage("--expect needs a value"))?;
                options.expect = Some(value.clone());
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
        "{detail}\nusage: h3_vision_gate <FL2VA/text_encoder> <image> \
         [--height H] [--width W] [--expect HEX]"
    )
}
