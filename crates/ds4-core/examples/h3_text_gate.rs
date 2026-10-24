//! The E4 gate: the port's prompt encoder against the reference's golden.
//!
//! Usage: h3_text_gate <FL2VA/text_encoder> <tokenizer.json> <prompt>
//!                     [--layers N] [--expect HEX]
//!
//! Prints the prompt ids, then one PARITY line with the shape, the FNV-1a hash
//! over the BF16 payload, both edges and the wall time. With `--expect` the
//! process exits 3 when the hash differs.
//!
//! The golden this gate was written against, recorded on the Spark with the
//! reference encoder (`tools/minimax_h3/text_probe.c`, commit 021fa29d):
//!
//!   TOKENS 32 2518 38835 11435 1526 11794
//!   PARITY tokens=6 width=5120 values=30720 fnv=1d9d71e8dd8544d6
//!          first=3f99.bf71.bfc7.c054.4090.c06f.3f1b.c083
//!          last=bef4.c066.c040.3db8.3d9e.3fbe.bd80.4000

use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::text_forward::{self, Embedding};
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

    if positional.len() != 3 {
        return Err(usage("expected three positional arguments"));
    }

    let encoder = &positional[0];
    let tokenizer_path = &positional[1];
    let prompt = &positional[2];
    let layers = match options.layers {
        Some(layers) if layers == 0 || layers > ds4_core::minimax_h3::text::LAYERS => {
            return Err(usage("--layers is out of range"));
        }
        Some(layers) => layers,
        None => ds4_core::minimax_h3::text::LAYERS,
    };
    let expect = match options.expect.as_deref() {
        Some(text) => {
            Some(parse_hash(text).ok_or_else(|| usage("--expect needs a hexadecimal hash"))?)
        }
        None => None,
    };

    let tokenizer = Tokenizer::load(std::path::Path::new(tokenizer_path))
        .map_err(|error| format!("tokenizer: {}", error.token()))?;
    let ids = tokenizer
        .encode(prompt, EmptyPrompt::Pad)
        .map_err(|error| format!("tokenizer: {}", error.token()))?;

    println!(
        "TOKENS {}",
        ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
    );

    let start = Instant::now();
    let embedding = text_forward::encode_layers(std::path::Path::new(encoder), &ids, layers)
        .map_err(|error| error.token())?;
    let seconds = start.elapsed().as_secs_f64();

    println!(
        "{} layers={layers} seconds={seconds:.3}",
        parity(&embedding)
    );

    if let Some(expect) = expect {
        if embedding.fnv() != expect {
            return Ok(ExitCode::from(3));
        }
    }

    Ok(ExitCode::SUCCESS)
}

/// The one line the plan records, in the probe's spelling.
fn parity(embedding: &Embedding) -> String {
    format!(
        "PARITY tokens={} width={} values={} fnv={:016x} first={} last={}",
        embedding.tokens(),
        embedding.width(),
        embedding.tokens() * embedding.width(),
        embedding.fnv(),
        words(&embedding.first_words(8)),
        words(&embedding.last_words(8))
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

/// The three positional arguments, plus the two flags the gate knows.
struct Options {
    layers: Option<u32>,
    expect: Option<String>,
}

fn split(args: &[String]) -> Result<(Vec<String>, Options), String> {
    let mut positional = Vec::new();
    let mut options = Options {
        layers: None,
        expect: None,
    };
    let mut index = 0;

    while index < args.len() {
        let argument = &args[index];

        match argument.as_str() {
            "--layers" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| usage("--layers needs a value"))?;
                options.layers = Some(
                    value
                        .parse::<u32>()
                        .map_err(|_| usage("--layers needs a number"))?,
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
        "{detail}\nusage: h3_text_gate <FL2VA/text_encoder> <tokenizer.json> <prompt> \
         [--layers N] [--expect HEX]"
    )
}
