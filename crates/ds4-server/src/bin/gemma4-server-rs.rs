//! An OpenAI-compatible host for a Gemma-4-26B-A4B VQ artifact, on the ds4
//! server's own wire host.
//!
//! The routes, the request loop, the sampler face, streaming and the reply
//! shape are `ds4_server::accept_loop_with_engine` — the same host every
//! other family serves through. This binary only builds the family engine
//! (`gemma4_engine`, over the device chain in `native/gemma4/gemma4_gpu.cu`)
//! and hands it to that loop.
//!
//! `POST /v1/chat/completions` (buffered and streamed), `POST /v1/completions`
//! and `GET /v1/models` come from the shared host; the prompt is the
//! checkpoint's own Gemma-4 template, and the reply parser is the server's.

mod gemma4_engine;

use std::net::TcpListener;
use std::path::PathBuf;
use std::process;

use ds4_core::gemma4_a4b::Tokenizer;
use ds4_server::{accept_loop_with_engine, ServerConfig};

use gemma4_engine::{Gemma4Engine, GpuModel};

const DEFAULT_BIND: &str = "127.0.0.1:8080";
const DEFAULT_NAME: &str = "gemma-4-26b-a4b-vq";

struct Args {
    bind: String,
    dir: PathBuf,
    tokenizer: PathBuf,
    name: String,
    ctx: i32,
    max_tokens: i32,
}

fn usage() -> &'static str {
    "gemma4-server-rs - an OpenAI-compatible host for a Gemma-4 VQ artifact

Usage: gemma4-server-rs --dir DIR --tokenizer FILE [options]

  --bind ADDR:PORT    listen address (default 127.0.0.1:8080)
  --dir DIR           artifact directory holding vq-release.json
  --tokenizer FILE    the checkpoint's tokenizer.json
  --model-dir DIR     checkpoint root (tokenizer.json is taken from it)
  --name NAME         model id reported by the API (default gemma-4-26b-a4b-vq)
  --ctx N             context limit for a request (default 8192)
  --max-tokens N      default completion length (default 128)
  --help              this text"
}

fn parse_args() -> Args {
    let mut bind = DEFAULT_BIND.to_string();
    let mut dir = None;
    let mut tokenizer = None;
    let mut model_dir = None;
    let mut name = DEFAULT_NAME.to_string();
    let mut ctx = 8192;
    let mut max_tokens = 128;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bind" => bind = args.next().unwrap_or(bind),
            "--dir" => dir = args.next(),
            "--tokenizer" => tokenizer = args.next(),
            "--model-dir" => model_dir = args.next(),
            "--name" => name = args.next().unwrap_or(name),
            "--ctx" => ctx = args.next().and_then(|v| v.parse().ok()).unwrap_or(ctx),
            "--max-tokens" => {
                max_tokens = args.next().and_then(|v| v.parse().ok()).unwrap_or(max_tokens)
            }
            "--help" | "-h" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => {
                eprintln!("unknown argument {other}\n\n{}", usage());
                process::exit(2);
            }
        }
    }
    let Some(dir) = dir else {
        eprintln!("--dir is required\n\n{}", usage());
        process::exit(2);
    };
    let tokenizer = tokenizer
        .or_else(|| {
            model_dir.map(|dir| format!("{}/tokenizer.json", dir.trim_end_matches('/')))
        })
        .unwrap_or_else(|| {
            eprintln!("--tokenizer (or --model-dir) is required\n\n{}", usage());
            process::exit(2);
        });
    Args {
        bind,
        dir: dir.into(),
        tokenizer: tokenizer.into(),
        name,
        ctx,
        max_tokens,
    }
}

fn main() {
    let args = parse_args();
    let tok = match Tokenizer::open(&args.tokenizer) {
        Ok(tok) => tok,
        Err(e) => {
            eprintln!("tokenizer {}: {e}", args.tokenizer.display());
            process::exit(1);
        }
    };
    eprintln!("opening {}", args.dir.display());
    let model = match GpuModel::open(&args.dir, args.ctx.max(1) as usize) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("open: {e}");
            process::exit(1);
        }
    };
    let ctx = args.ctx;
    let mut engine = Gemma4Engine::new(model, tok, ctx);

    let (host, port) = match args.bind.rsplit_once(':') {
        Some((host, port)) => (host.to_string(), port.parse::<u16>().unwrap_or(8080)),
        None => (args.bind.clone(), 8080),
    };
    let mut cfg = ServerConfig::default();
    cfg.listen_host = host;
    cfg.listen_port = port;
    cfg.model_id = args.name.clone();
    cfg.model_name = args.name;
    cfg.ctx = ctx;
    cfg.default_tokens = args.max_tokens;
    cfg.have_engine = true;
    // The engine's own request defaults: thinking OFF unless a request asks
    // for it (parse.rs ParseEnv, server_msgs.c request_init). The Gemma-4
    // template closes an empty thought channel in that case, and the reply
    // parser only has a channel entry for the families it knows, so a reply
    // that thinks would be reported as reasoning with its markers in the
    // text. With thinking off the answer arrives on the content channel,
    // which is the path the gate's "Stockholm" greedy run exercises.
    cfg.engine_defaults = true;

    let listener = match TcpListener::bind((cfg.listen_host.as_str(), cfg.listen_port)) {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("bind {}:{}: {e}", cfg.listen_host, cfg.listen_port);
            process::exit(1);
        }
    };
    eprintln!(
        "serving {} on http://{}:{}",
        cfg.model_id, cfg.listen_host, cfg.listen_port
    );
    accept_loop_with_engine(listener, cfg, &mut engine);
}
