//! Standalone OpenAI Videos API host for the MiniMax-H3 (h3) family.
//!
//! No GPU is opened at startup: generation is delegated to the operator's
//! `--renderer` program. Without one, the host still starts and every
//! `POST /v1/videos` answers 503 in the OpenAI error body, which is how a
//! deployment can be probed before the P8 renderer lands.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process;
use std::sync::Arc;
use std::time::Duration;

use ds4_core::minimax_h3::video_api::{VideoStore, DEFAULT_MODEL};
use ds4_server::h3_video::{serve, CommandRenderer, DisabledRenderer, SharedRenderer, VideoServer};

const DEFAULT_BIND: &str = "127.0.0.1:8090";
const DEFAULT_OUT_DIR: &str = "h3-videos";
const DEFAULT_MAX_JOBS: usize = 8;
const DEFAULT_TIMEOUT_SECS: u64 = 3600;

struct Args {
    bind: String,
    model: String,
    out_dir: PathBuf,
    renderer: Option<PathBuf>,
    renderer_args: Vec<String>,
    max_jobs: usize,
    timeout: Duration,
}

fn usage() -> &'static str {
    "h3-server-rs - OpenAI Videos API for MiniMax-H3

Usage: h3-server-rs [options]

  --bind ADDR:PORT       listen address (default 127.0.0.1:8090)
  --model ID             model id accepted by POST /v1/videos
                         (default minimax-h3)
  --out-dir DIR          directory for rendered MP4 files (default h3-videos)
  --renderer PROGRAM     external renderer executable; without one every
                         POST /v1/videos answers 503
  --renderer-arg ARG     fixed argument passed to the renderer (repeatable)
  --max-jobs N           retained jobs, also render concurrency (default 8)
  --timeout-secs N       per-render timeout in seconds (default 3600)
  --help                 this text

The renderer is invoked with the reference engine's own CLI spellings:

  PROGRAM [fixed args...] -p P --width W --height H --seconds S -o PATH

A fixed model directory is supplied with two renderer args, for example:

  --renderer /home/leandro/h3-oracle/h3 \\
  --renderer-arg -d --renderer-arg /mnt/h3-artifacts/MiniMax-H3

with no shell. The program, its fixed arguments, the output directory and the
working directory are chosen here, never by a request body. The P8 port keeps
the same CLI flags (port-plan 6.3), so the renderer can be swapped without
changing this surface.
"
}

fn parse_args() -> Args {
    let mut args = Args {
        bind: DEFAULT_BIND.to_string(),
        model: DEFAULT_MODEL.to_string(),
        out_dir: PathBuf::from(DEFAULT_OUT_DIR),
        renderer: None,
        renderer_args: Vec::new(),
        max_jobs: DEFAULT_MAX_JOBS,
        timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                print!("{}", usage());
                process::exit(0);
            }
            "--bind" => args.bind = value(&mut argv, &flag),
            "--model" => args.model = value(&mut argv, &flag),
            "--out-dir" => args.out_dir = PathBuf::from(value(&mut argv, &flag)),
            "--renderer" => args.renderer = Some(PathBuf::from(value(&mut argv, &flag))),
            "--renderer-arg" => args.renderer_args.push(value(&mut argv, &flag)),
            "--max-jobs" => {
                let text = value(&mut argv, &flag);
                args.max_jobs = text.parse().unwrap_or_else(|_| fail(&flag, &text));
            }
            "--timeout-secs" => {
                let text = value(&mut argv, &flag);
                let secs: u64 = text.parse().unwrap_or_else(|_| fail(&flag, &text));
                args.timeout = Duration::from_secs(secs.max(1));
            }
            other => {
                eprintln!("h3-server-rs: unknown option {other}\n");
                eprint!("{}", usage());
                process::exit(2);
            }
        }
    }
    args
}

fn value(argv: &mut impl Iterator<Item = String>, flag: &str) -> String {
    argv.next().unwrap_or_else(|| {
        eprintln!("h3-server-rs: {flag} needs a value\n");
        eprint!("{}", usage());
        process::exit(2);
    })
}

fn fail(flag: &str, text: &str) -> ! {
    eprintln!("h3-server-rs: invalid value for {flag}: {text}");
    process::exit(2);
}

fn main() {
    let args = parse_args();
    if let Err(error) = std::fs::create_dir_all(&args.out_dir) {
        eprintln!(
            "h3-server-rs: cannot create output directory {}: {error}",
            args.out_dir.display()
        );
        process::exit(1);
    }

    let store = Arc::new(VideoStore::new(args.model.clone(), args.max_jobs));
    let renderer: SharedRenderer = match &args.renderer {
        Some(program) => Arc::new(CommandRenderer::new(
            program.clone(),
            args.renderer_args.clone(),
            args.timeout,
        )),
        None => Arc::new(DisabledRenderer),
    };
    let server = Arc::new(VideoServer::new(store, renderer, args.out_dir.clone()));
    let listener = match TcpListener::bind(&args.bind) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("h3-server-rs: cannot bind {}: {error}", args.bind);
            process::exit(1);
        }
    };
    let address = listener
        .local_addr()
        .map(|address| address.to_string())
        .unwrap_or_else(|_| args.bind.clone());
    println!("h3-server-rs: OpenAI Videos API on http://{address}/v1");
    println!("  model {}", args.model);
    println!("  output {}", args.out_dir.display());
    match &args.renderer {
        Some(program) => println!(
            "  renderer {} ({} fixed args, {}s timeout)",
            program.display(),
            args.renderer_args.len(),
            args.timeout.as_secs()
        ),
        None => println!("  renderer disabled (POST /v1/videos answers 503)"),
    }
    if let Err(error) = serve(listener, server) {
        eprintln!("h3-server-rs: server stopped: {error}");
        process::exit(1);
    }
}
