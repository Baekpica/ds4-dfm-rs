//! Run the Gemma-4-26B-A4B VQ artifact on the CPU reference forward.
//!
//! ```
//! cargo run --release --example gemma4_a4b_run -- \
//!     --dir /data/vq-artifacts/gemma-4-26b-a4b-8x4096 \
//!     --tokenizer /data/models/gemma-4-26B-A4B-it/tokenizer.json \
//!     --prompt "Hello" --generate 8
//! ```
//!
//! It prints the prompt ids, each produced token with its id and the wall
//! time of the step, and the decoded reply. The forward is one token at a
//! time and unoptimised; the timings are the shape of the work, not a
//! performance claim.

use std::process;
use std::time::Instant;

use ds4_core::gemma4_a4b::{Model, Tokenizer};

fn usage() -> &'static str {
    "gemma4_a4b_run - the CPU reference forward over a vq-release/1 artifact

  --dir DIR          artifact directory (vq-release.json)
  --tokenizer FILE   the checkpoint's tokenizer.json
  --model-dir DIR    checkpoint root (tokenizer.json is taken from it)
  --prompt TEXT      prompt text (default: \"Hello\")
  --generate N       tokens to generate (default 8)
  --raw              do not wrap the prompt in the chat template
  --help             this text"
}

fn main() {
    let mut dir = None;
    let mut tokenizer = None;
    let mut model_dir = None;
    let mut prompt = String::from("Hello");
    let mut generate = 8usize;
    let mut raw = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dir" => dir = args.next(),
            "--tokenizer" => tokenizer = args.next(),
            "--model-dir" => model_dir = args.next(),
            "--prompt" => prompt = args.next().unwrap_or_default(),
            "--generate" => generate = args.next().and_then(|v| v.parse().ok()).unwrap_or(8),
            "--raw" => raw = true,
            "--help" | "-h" => {
                println!("{}", usage());
                return;
            }
            other => {
                eprintln!("unknown argument {other}");
                process::exit(2);
            }
        }
    }
    let Some(dir) = dir else {
        eprintln!("--dir is required\n\n{}", usage());
        process::exit(2);
    };
    let tokenizer = tokenizer.map(Into::into).or_else(|| {
        model_dir.map(|root| format!("{root}/tokenizer.json"))
    });
    let Some(tokenizer) = tokenizer else {
        eprintln!("--tokenizer or --model-dir is required\n\n{}", usage());
        process::exit(2);
    };

    let tok = match Tokenizer::open(std::path::Path::new(&tokenizer)) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("tokenizer: {e}");
            process::exit(1);
        }
    };

    let started = Instant::now();
    let model = match Model::open(std::path::Path::new(&dir)) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("open: {e}");
            process::exit(1);
        }
    };
    println!("opened in {:.1} s", started.elapsed().as_secs_f64());

    let ids: Vec<u32> = if raw {
        tok.encode(&prompt)
    } else {
        match tok.chat_prompt(&[("user".into(), prompt.clone())], false) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("template: {e}");
                process::exit(1);
            }
        }
    };
    println!("prompt ids: {ids:?}");
    println!("prompt text: {:?}", tok.decode(&ids));

    let mut session = model.session();
    let mut next = Vec::new();
    let mut produced = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        let at = Instant::now();
        next = match session.step(id) {
            Ok(logits) => logits,
            Err(e) => {
                eprintln!("step {i}: {e}");
                process::exit(1);
            }
        };
        println!(
            "prompt {i:>3} id {id:>6} {:>8.2} s  {:?}",
            at.elapsed().as_secs_f64(),
            tok.token(id).unwrap_or("<special>")
        );
    }
    for i in 0..generate {
        let best = next
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id as u32)
            .unwrap_or(0);
        produced.push(best);
        if tok.is_eos(best) {
            println!("generated {i:>3} id {best:>6} <eos>");
            break;
        }
        let at = Instant::now();
        next = match session.step(best) {
            Ok(logits) => logits,
            Err(e) => {
                eprintln!("step {i}: {e}");
                process::exit(1);
            }
        };
        println!(
            "generated {i:>3} id {best:>6} {:>8.2} s  {:?}",
            at.elapsed().as_secs_f64(),
            tok.token(best).unwrap_or("<special>")
        );
    }
    println!("reply ids: {produced:?}");
    println!("reply: {:?}", tok.decode(&produced));
}
