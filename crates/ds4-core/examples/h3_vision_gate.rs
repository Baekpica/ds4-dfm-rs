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
//! Nothing about the reference's own probe is assumed here: the golden is what
//! `~/h3-oracle` prints for the same image at the same shape, recorded in this
//! example's doc comment once measured on the Spark.

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
    format!(
        "{label} tokens={} width={} values={} fnv={hash:016x} first={} last={}",
        output.tokens(),
        output.width(),
        payload.len() / 2,
        words(&output.first_words(8)),
        words(&output.last_words(8))
    )
}

fn words(values: &[u16]) -> String {
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
