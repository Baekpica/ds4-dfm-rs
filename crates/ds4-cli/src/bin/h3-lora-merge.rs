//! `h3-lora-merge` — merge the FastH3 distilled adapter into a derived
//! MiniMax-H3 checkpoint.
//!
//! The reference's stage 1 merges this adapter before its FP8 conversion and
//! then needs four denoise evaluations instead of the base model's twenty
//! (`Sol-H3-Spark/configs/default.json:6-9`). The merge is offline by design:
//! the inference path keeps loading one weight format, the merged checkpoint is
//! an artifact that can be hashed and reused, and a wrong merge is a directory
//! to delete instead of a kernel bug.
//!
//! Usage:
//!
//!     h3-lora-merge --model <MiniMax-H3> --adapter <adapter_model.safetensors>
//!                   --out <derived directory> [--threads N] [--check]
//!
//! The released `--model` directory is only ever read. `--check` resolves every
//! adapter key against it, prints the plan and writes nothing.

use std::path::PathBuf;
use std::process::ExitCode;

use ds4_core::minimax_h3::layout::Config;
use ds4_core::minimax_h3::lora::{self, Adapter, Delta, Plan, Rows};
use ds4_core::minimax_h3::safetensors;

const USAGE: &str = "\
usage: h3-lora-merge --model <MiniMax-H3> --adapter <adapter_model.safetensors> \
--out <directory> [--threads N] [--check]

  --model    the released checkpoint directory (read only)
  --adapter  a low-rank adapter: the FastH3 distilled one (fastvideo-lora-v2)
             or a bare one that names the served tensors, such as
             larryvrh/MiniMax-H3-Turbo-Lora
  --out      where the derived checkpoint goes; must not exist
  --threads  worker threads for the low-rank products (default: available parallelism)
  --check    resolve and report the plan, write nothing
  --help     this text";

struct Args {
    model: PathBuf,
    adapter: PathBuf,
    out: Option<PathBuf>,
    threads: usize,
    check: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("h3-lora-merge: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse(std::env::args().skip(1).collect())?;

    if args.check {
        return check(&args);
    }

    let out = args.out.ok_or("--out is required")?;
    let report = lora::derive(&args.model, &args.adapter, &out, args.threads, false)
        .map_err(|error| error.to_string())?;

    println!("{report}");
    println!("derived: {}", out.display());
    Ok(())
}

fn parse(argv: Vec<String>) -> Result<Args, String> {
    let mut model: Option<PathBuf> = None;
    let mut adapter: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let mut check = false;

    let mut index = 0;
    while index < argv.len() {
        let flag = argv[index].clone();
        let value = |index: usize, flag: &str| -> Result<String, String> {
            argv.get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };

        match flag.as_str() {
            "--model" => {
                model = Some(PathBuf::from(value(index, &flag)?));
                index += 2;
            }
            "--adapter" => {
                adapter = Some(PathBuf::from(value(index, &flag)?));
                index += 2;
            }
            "--out" => {
                out = Some(PathBuf::from(value(index, &flag)?));
                index += 2;
            }
            "--threads" => {
                threads = value(index, &flag)?
                    .parse()
                    .map_err(|_| "--threads is not a number".to_string())?;
                if threads == 0 {
                    return Err("--threads must be at least 1".into());
                }
                index += 2;
            }
            "--check" => {
                check = true;
                index += 1;
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}\n\n{USAGE}")),
        }
    }

    Ok(Args {
        model: model.ok_or("--model is required")?,
        adapter: adapter.ok_or("--adapter is required")?,
        out,
        threads,
        check,
    })
}

/// `--check`: the plan, without writing a byte.
fn check(args: &Args) -> Result<(), String> {
    let source = args.model.join(lora::PARTITION).join("transformer");
    let config = Config::read(&source).map_err(|error| error.token())?;
    let adapter = Adapter::open(&args.adapter).map_err(|error| error.to_string())?;
    let store = safetensors::open_dir(&source).map_err(|error| error.to_string())?;
    let plan = Plan::build(&adapter, &store, &config).map_err(|error| error.to_string())?;

    println!("adapter {}", args.adapter.display());
    for key in [
        "format",
        "rank",
        "base_revision",
        "base_model",
        "application",
    ] {
        if let Some(value) = adapter.shard().metadata.get(key) {
            println!("  {key}: {value}");
        }
    }

    println!(
        "served checkpoint: {} tensors, {} shards",
        store.tensor_count(),
        store.shards().len()
    );
    println!(
        "plan: {} tensors touched, {} low-rank modules, {} additive keys, {} whole-parameter modules",
        plan.touches().len(),
        plan.low_rank_modules(),
        plan.additive_keys(),
        plan.whole().len()
    );

    for (name, deltas) in plan.touches() {
        let (_, tensor) = store.find(name).expect("a planned tensor is in the store");
        let (rows, cols) = if tensor.shape.len() == 1 {
            (1, tensor.shape[0])
        } else {
            (tensor.shape[0], tensor.shape[1..].iter().product())
        };

        let kinds = deltas
            .iter()
            .map(|delta| match delta {
                Delta::Add { .. } => "add".to_string(),
                Delta::LowRank { rows, .. } => match rows {
                    Rows::Identity => "low-rank".to_string(),
                    Rows::Qkv { part } => format!("low-rank[qkv {part}]"),
                    Rows::Swap => "low-rank[swap]".to_string(),
                },
            })
            .collect::<Vec<_>>()
            .join("+");

        println!("  {name}  {rows}x{cols}  {kinds}");
    }

    for module in plan.whole() {
        println!("  {module}.weight  whole (the released checkpoint does not carry it)");
    }

    Ok(())
}
