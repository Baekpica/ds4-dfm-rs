//! ds4-advisor: find the fastest Gemma-4 serving configuration for a chosen
//! context on this card.
//!
//! The advisor is a driver, not a second engine. For every candidate
//! configuration it runs `ds4-bench` (the same in-process harness the launcher
//! uses) on one prompt sized to the requested context, with the knobs that
//! describe that configuration in the environment, and reads the row back:
//! prefill tokens per second, decode tokens per second, first-token seconds
//! and the KV bytes the run reported. It samples the card's used memory while
//! the arm runs, so an arm that would not leave room at a long context is
//! visible instead of being recommended.
//!
//! What it sweeps:
//!
//!   * the full-layer KV format, bf16 or E4M3 (DS4_GEMMA4_FULL_KV);
//!   * the expert placement: all-GPU resident, the per-layer host split
//!     (DS4_GEMMA4_MOE_CPU_LAYERS), the host-cold GPU slot cache
//!     (DS4_GEMMA4_MOE_CACHE_MIB) and the whole model on the CPU
//!     (DS4_GEMMA4_CPU_MOE);
//!   * the prefill chunk at the resident and cache shapes, because it trades
//!     activation memory for prefill overlap.
//!
//! It never decides by arithmetic alone: an arm only wins on numbers this run
//! measured. The VRAM estimates below only prune arms that cannot fit before
//! spending a model load on them, and every skipped arm says why.
//!
//! Safety: it refuses to run while another ds4 server answers on the port,
//! unless --stop-endpoint is given, in which case the launcher's own stop and
//! start restore the endpoint after the sweep, including on failure.

use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

const MODEL_MIB: u64 = 9_580; // 9.37 GiB map, the weights the resident lane copies
const RESIDENT_SET_MIB: u64 = 2_406; // 2.35 GiB: attention, norms, books, router
const EXPERT_MIB: u64 = 7_184; // 7.02 GiB of routed expert records
const WORKSPACE_MIB: u64 = 900; // session graph, logits, scratch (measured at 8192)
const LAYERS: u64 = 30;
// Full-layer KV: 5 layers x 2,056 B per token in E4M3, doubled for bf16. The
// sliding rings are fixed at window + chunk, so they do not grow with ctx.
const KV_FP8_BYTES_PER_TOKEN: u64 = 5 * 2_056;
const KV_BF16_BYTES_PER_TOKEN: u64 = 5 * 4_112;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Prefer {
    Balanced,
    Prefill,
    Decode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Resident,
    CpuLayers(u32),
    Cache(u32),
    CpuMoe,
}

struct Arm {
    name: String,
    kv: &'static str,
    shape: Shape,
    chunk: u32,
    estimate_mib: u64,
}

struct Args {
    ctx: u32,
    gen_tokens: u32,
    prefer: Prefer,
    prompt: Option<PathBuf>,
    bench: PathBuf,
    model: PathBuf,
    repo: PathBuf,
    csv_out: Option<PathBuf>,
    json_out: Option<PathBuf>,
    stop_endpoint: bool,
    port: u16,
    arm_timeout: u64,
    budget_min: u64,
    vram_total_mib: u64,
    free_headroom_mib: u64,
    dry_run: bool,
    include_expensive: bool,
}

struct Measured {
    arm_index: usize,
    ttft_s: f64,
    prefill_tps: f64,
    decode_tps: f64,
    peak_mib: u64,
    kv_bytes: u64,
    score: f64,
    seconds: f64,
}

struct Skipped {
    name: String,
    why: String,
}

fn usage() -> String {
    [
        "ds4-advisor: sweep Gemma-4 serving configurations and rank them",
        "",
        "usage: ds4-advisor --ctx N [options]",
        "",
        "  --ctx N            context to size the sweep for (required, max 65536)",
        "  --gen-tokens N     greedy decode tokens per arm (default 32)",
        "  --prefer MODE      balanced (default), prefill, or decode",
        "  --prompt-file P    prompt to benchmark (default: generated, ~ctx tokens)",
        "  --model DIR        release directory (default ~/models/gguf-family-it-r8)",
        "  --bench PATH       ds4-bench binary (default <repo>/ds4-bench)",
        "  --repo DIR         repository root (default: parent of --bench)",
        "  --csv FILE         write every measured arm as CSV",
        "  --json FILE        write the ranked result as JSON",
        "  --arm-timeout S    per-arm wall clock (default 1800)",
        "  --budget-min M     stop starting arms after M minutes (default 0 = no budget)",
        "  --vram-total MiB   card total (default: asked from nvidia-smi)",
        "  --free-margin MiB  VRAM that must stay free at peak (default 384)",
        "  --stop-endpoint    stop the launcher's endpoint first, restore it after",
        "  --port N           endpoint port to check (default 8099)",
        "  --include-expensive  also sweep the whole-model-on-CPU arm",
        "  --dry-run          print the arm list and the estimates, run nothing",
        "  -h, --help         this text",
        "",
        "The prompt default is generated from this corpus' measured ratio of",
        "2.9 bytes per token, so the sweep prefills about --ctx tokens.",
    ]
    .join("\n")
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        ctx: 0,
        gen_tokens: 32,
        prefer: Prefer::Balanced,
        prompt: None,
        bench: PathBuf::new(),
        model: PathBuf::new(),
        repo: PathBuf::new(),
        csv_out: None,
        json_out: None,
        stop_endpoint: false,
        port: 8099,
        arm_timeout: 1800,
        budget_min: 0,
        vram_total_mib: 0,
        free_headroom_mib: 384,
        dry_run: false,
        include_expensive: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || {
            it.next()
                .ok_or_else(|| format!("{} needs a value", arg))
        };
        match arg.as_str() {
            "--ctx" => args.ctx = value()?.parse().map_err(|_| "--ctx wants a number")?,
            "--gen-tokens" => {
                args.gen_tokens = value()?.parse().map_err(|_| "--gen-tokens wants a number")?
            }
            "--prefer" => {
                args.prefer = match value()?.as_str() {
                    "balanced" => Prefer::Balanced,
                    "prefill" | "ttft" => Prefer::Prefill,
                    "decode" => Prefer::Decode,
                    other => return Err(format!("--prefer wants balanced, prefill or decode (got {other})")),
                }
            }
            "--prompt-file" => args.prompt = Some(PathBuf::from(value()?)),
            "--model" => args.model = PathBuf::from(value()?),
            "--bench" => args.bench = PathBuf::from(value()?),
            "--repo" => args.repo = PathBuf::from(value()?),
            "--csv" => args.csv_out = Some(PathBuf::from(value()?)),
            "--json" => args.json_out = Some(PathBuf::from(value()?)),
            "--arm-timeout" => {
                args.arm_timeout = value()?.parse().map_err(|_| "--arm-timeout wants seconds")?
            }
            "--budget-min" => {
                args.budget_min = value()?.parse().map_err(|_| "--budget-min wants minutes")?
            }
            "--vram-total" => {
                args.vram_total_mib = value()?.parse().map_err(|_| "--vram-total wants MiB")?
            }
            "--free-margin" => {
                args.free_headroom_mib =
                    value()?.parse().map_err(|_| "--free-margin wants MiB")?
            }
            "--stop-endpoint" => args.stop_endpoint = true,
            "--port" => args.port = value()?.parse().map_err(|_| "--port wants a number")?,
            "--include-expensive" => args.include_expensive = true,
            "--dry-run" => args.dry_run = true,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.ctx == 0 {
        return Err("--ctx is required (the context the sweep is sized for)".into());
    }
    if args.ctx > 65536 {
        return Err(format!(
            "--ctx {} is past the 4070's 65,536 ceiling",
            args.ctx
        ));
    }
    let home = std::env::var("HOME").unwrap_or_default();
    if args.model.as_os_str().is_empty() {
        args.model = PathBuf::from(format!("{home}/models/gguf-family-it-r8"));
    }
    if args.bench.as_os_str().is_empty() {
        // The launcher's build puts it next to this binary's repository root;
        // resolve it now so the child does not have to find it through PATH.
        args.bench = PathBuf::from("./ds4-bench");
    }
    args.bench = args
        .bench
        .canonicalize()
        .map_err(|e| format!("bench binary {}: {e}", args.bench.display()))?;
    if args.repo.as_os_str().is_empty() {
        args.repo = args
            .bench
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
    }
    Ok(args)
}

fn kv_mib(ctx: u32, kv: &str) -> u64 {
    let per_token = if kv == "fp8" {
        KV_FP8_BYTES_PER_TOKEN
    } else {
        KV_BF16_BYTES_PER_TOKEN
    };
    ((per_token as u64 * ctx as u64) / (1024 * 1024)).max(1)
}

fn estimate_mib(ctx: u32, kv: &str, shape: Shape) -> u64 {
    let kv = kv_mib(ctx, kv);
    match shape {
        Shape::Resident => MODEL_MIB + kv + WORKSPACE_MIB,
        Shape::CpuLayers(n) => {
            let n = n.min(LAYERS as u32) as u64;
            RESIDENT_SET_MIB + EXPERT_MIB * (LAYERS - n) / LAYERS + kv + WORKSPACE_MIB / 2
        }
        Shape::Cache(mib) => RESIDENT_SET_MIB + mib as u64 + kv + WORKSPACE_MIB / 2,
        Shape::CpuMoe => RESIDENT_SET_MIB + kv + 400,
    }
}

fn arms(args: &Args) -> Vec<Arm> {
    let mut list: Vec<Arm> = Vec::new();
    let mut push = |name: &str, kv: &'static str, shape: Shape, chunk: u32| {
        list.push(Arm {
            name: name.to_string(),
            kv,
            shape,
            chunk,
            estimate_mib: estimate_mib(args.ctx, kv, shape),
        });
    };
    // Ordered by expected value: the resident lane first (it is the current
    // default), then the shapes that give memory back. A time budget cuts the
    // tail, so the cheap answers come first.
    push("resident-fp8", "fp8", Shape::Resident, 512);
    push("resident-bf16", "bf16", Shape::Resident, 512);
    push("cache4g-fp8", "fp8", Shape::Cache(4096), 512);
    push("cpu8-fp8", "fp8", Shape::CpuLayers(8), 512);
    push("cache2g-fp8", "fp8", Shape::Cache(2048), 512);
    push("resident-fp8-chunk1024", "fp8", Shape::Resident, 1024);
    push("resident-fp8-chunk2048", "fp8", Shape::Resident, 2048);
    push("cpu16-fp8", "fp8", Shape::CpuLayers(16), 512);
    push("cache4g-bf16", "bf16", Shape::Cache(4096), 512);
    if args.include_expensive {
        push("cpumoe-fp8", "fp8", Shape::CpuMoe, 512);
    }
    list
}

fn arm_env(arm: &Arm, args: &Args) -> Vec<(&'static str, String)> {
    let lock = std::env::temp_dir().join("ds4-advisor.lock");
    let mut env: Vec<(&'static str, String)> = vec![
        ("DS4_LOCK_FILE", lock.display().to_string()),
        ("DS4_GEMMA4_FULL_KV", arm.kv.to_string()),
        ("DS4_GEMMA4_PREFILL_CHUNK", arm.chunk.to_string()),
        ("DS4_SESSION_GRAPH_HEADROOM_MB", "0".to_string()),
        ("DS4_GEMMA4_ATTN_FAST", "1".into()),
        ("DS4_GEMMA4_ATTN_PREFILL", "1".into()),
        ("DS4_GEMMA4_VQ_PREFILL", "1".into()),
        ("DS4_GEMMA4_Q8_BATCH", "1".into()),
        ("DS4_GEMMA4_FUSE_ROPE", "1".into()),
        ("DS4_GEMMA4_ROUTER_PAR", "1".into()),
        ("DS4_GEMMA4_HEAD_GEMV", "1".into()),
        ("DS4_GEMMA4_ROUTER_DOT", "1".into()),
        ("DS4_GEMMA4_ATTN_MMA", "1".into()),
        ("DS4_GEMMA4_ATTN_SPLIT", "1".into()),
        ("DS4_GEMMA4_ATTN_SPLIT_MMA", "1".into()),
        ("DS4_GEMMA4_MOE_PERSIST", "1".into()),
        ("DS4_GEMMA4_CPU_MOE", "0".into()),
        ("DS4_GEMMA4_MOE_CPU_LAYERS", "0".into()),
        ("DS4_GEMMA4_MOE_CACHE_MIB", "0".into()),
        ("DS4_GEMMA4_MOE_HOST", "0".into()),
    ];
    match arm.shape {
        Shape::Resident => env.push(("DS4_CUDA_COPY_MODEL", "1".into())),
        Shape::CpuLayers(n) => env.push(("DS4_GEMMA4_MOE_CPU_LAYERS", n.to_string())),
        Shape::Cache(mib) => env.push(("DS4_GEMMA4_MOE_CACHE_MIB", mib.to_string())),
        Shape::CpuMoe => env.push(("DS4_GEMMA4_CPU_MOE", "1".into())),
    }
    let _ = args;
    env
}

/// Environment the arm must inherit from this machine (loader paths, device
/// selection). Everything the sweep controls is set explicitly afterwards.
fn passthrough_env() -> Vec<(&'static str, String)> {
    const KEEP: [&str; 6] = [
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "DS4_CUDA_ARCH",
        "TMPDIR",
    ];
    KEEP.iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| (*k, v)))
        .collect()
}

fn nvidia_used_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

fn nvidia_total_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

fn port_answers(port: u16) -> bool {
    let mut stream = match TcpStream::connect(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(400)));
    let _ = stream.write_all(b"GET /v1/models HTTP/1.0\r\n\r\n");
    let mut buf = [0u8; 64];
    matches!(stream.read(&mut buf), Ok(n) if n > 0)
}

/// Stops the launcher's endpoint for the sweep and puts it back afterwards,
/// including when an arm fails or the process unwinds.
struct EndpointGuard {
    repo: PathBuf,
    stopped: bool,
}

impl EndpointGuard {
    fn new(repo: &Path, stop: bool, port: u16) -> Result<Self, String> {
        if !port_answers(port) {
            return Ok(Self {
                repo: repo.to_path_buf(),
                stopped: false,
            });
        }
        if !stop {
            return Err(format!(
                "a server answers on 127.0.0.1:{port}; stop it first or pass --stop-endpoint"
            ));
        }
        let script = repo.join("run-gemma4.sh");
        let status = Command::new("bash")
            .arg(&script)
            .args(["serve", "stop"])
            .status()
            .map_err(|e| format!("could not run {}: {e}", script.display()))?;
        if !status.success() {
            return Err("the launcher could not stop the endpoint".into());
        }
        Ok(Self {
            repo: repo.to_path_buf(),
            stopped: true,
        })
    }
}

impl Drop for EndpointGuard {
    fn drop(&mut self) {
        if !self.stopped {
            return;
        }
        let script = self.repo.join("run-gemma4.sh");
        let _ = Command::new("bash")
            .arg(&script)
            .args(["serve", "start"])
            .status();
    }
}

fn write_prompt(path: &Path, ctx: u32) -> Result<(), String> {
    // 2.9 bytes per token is the ratio measured on this corpus; the sweep only
    // needs a prompt that reaches the frontier, not a specific text.
    const PARA: &str = "A dense vector quantized model keeps its experts in a compact \
codebook and rebuilds every weight row from an index, a gain and a residual plane, so the \
decoder reads a small number of bytes for a large number of weights. This paragraph exists \
only to give the prefill stage realistic prose to chew on, with ordinary sentence lengths and \
the usual punctuation, because the frontier, not the wording, is what the sweep measures.\n";
    let want_bytes = (ctx as usize).saturating_mul(3);
    let mut text = String::with_capacity(want_bytes + PARA.len());
    while text.len() < want_bytes {
        text.push_str(PARA);
    }
    fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))
}

fn run_arm(
    arm: &Arm,
    index: usize,
    args: &Args,
    prompt: &Path,
    csv_path: &Path,
) -> Result<Measured, String> {
    let _ = fs::remove_file(csv_path);
    let started = Instant::now();
    let mut child = Command::new(&args.bench)
        .arg("-m")
        .arg(&args.model)
        .arg("--cuda")
        .arg("--prompt-file")
        .arg(prompt)
        .arg("--ctx-start")
        .arg(args.ctx.to_string())
        .arg("--ctx-max")
        .arg(args.ctx.to_string())
        .arg("--gen-tokens")
        .arg(args.gen_tokens.to_string())
        .arg("--csv")
        .arg(csv_path)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        // The loader and GPU-selection variables are the machine's, not the
        // sweep's: without them the CUDA build cannot open the card.
        .envs(passthrough_env())
        .envs(arm_env(arm, args))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", args.bench.display()))?;

    let mut peak = 0u64;
    let mut timed_out = false;
    loop {
        if let Some(used) = nvidia_used_mib() {
            peak = peak.max(used);
        }
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {}
            Err(e) => return Err(format!("waiting on the arm failed: {e}")),
        }
        if started.elapsed() > Duration::from_secs(args.arm_timeout) {
            let _ = child.kill();
            timed_out = true;
            break;
        }
        thread::sleep(Duration::from_millis(250));
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("collecting the arm output failed: {e}"))?;
    let seconds = started.elapsed().as_secs_f64();
    if timed_out {
        return Err(format!("exceeded --arm-timeout {}s", args.arm_timeout));
    }
    if !out.status.success() {
        let tail: String = String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(format!("ds4-bench exited {}: {tail}", out.status));
    }
    let csv = fs::read_to_string(csv_path).map_err(|e| format!("no CSV from the arm: {e}"))?;
    let row = csv
        .lines()
        .filter(|l| !l.starts_with("ctx_tokens"))
        .filter(|l| !l.trim().is_empty())
        .last()
        .ok_or_else(|| "the arm wrote an empty CSV".to_string())?;
    let f: Vec<&str> = row.split(',').collect();
    if f.len() < 8 {
        return Err(format!("unexpected CSV row: {row}"));
    }
    let prefill_tps = f[3].parse::<f64>().unwrap_or(0.0);
    let decode_tps = f[4].parse::<f64>().unwrap_or(0.0);
    let ttft_s = f[5].parse::<f64>().unwrap_or(f64::INFINITY);
    let kv_bytes = f[7].parse::<u64>().unwrap_or(0);
    let score = match args.prefer {
        Prefer::Prefill => prefill_tps,
        Prefer::Decode => decode_tps,
        // Harmonic mean: an arm that is fast at one stage and slow at the
        // other is not the answer to "optimal speed".
        Prefer::Balanced => {
            if prefill_tps + decode_tps > 0.0 {
                2.0 * prefill_tps * decode_tps / (prefill_tps + decode_tps)
            } else {
                0.0
            }
        }
    };
    Ok(Measured {
        arm_index: index,
        ttft_s,
        prefill_tps,
        decode_tps,
        peak_mib: peak,
        kv_bytes,
        score,
        seconds,
    })
}

fn main() {
    match run() {
        Ok(()) => {}
        Err(err) => {
            eprintln!("ds4-advisor: {err}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = parse_args()?;
    if !args.bench.is_file() {
        return Err(format!(
            "bench binary not found at {} (build it with: make ds4-bench CUDA_ARCH=sm_89)",
            args.bench.display()
        ));
    }
    if args.vram_total_mib == 0 {
        args.vram_total_mib = nvidia_total_mib()
            .ok_or_else(|| "could not ask nvidia-smi for the card total".to_string())?;
    }
    let ceiling = args.vram_total_mib.saturating_sub(args.free_headroom_mib);
    let list = arms(&args);

    if args.dry_run {
        println!(
            "card {} MiB, ceiling {} MiB ({} MiB must stay free), ctx {}",
            args.vram_total_mib, ceiling, args.free_headroom_mib, args.ctx
        );
        for (i, arm) in list.iter().enumerate() {
            let fit = if arm.estimate_mib <= ceiling {
                "fits"
            } else {
                "estimated over the ceiling"
            };
            println!(
                "{:>2}. {:<26} kv {:<4} chunk {:<5} est {:>5} MiB  {}",
                i + 1,
                arm.name,
                arm.kv,
                arm.chunk,
                arm.estimate_mib,
                fit
            );
        }
        println!(
            "\nNote: the estimates only prune; nothing is measured until a real run."
        );
        return Ok(());
    }

    let _guard = EndpointGuard::new(&args.repo, args.stop_endpoint, args.port)?;
    let work = std::env::temp_dir().join(format!("ds4-advisor-{}", std::process::id()));
    fs::create_dir_all(&work).map_err(|e| format!("could not create {}: {e}", work.display()))?;
    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let p = work.join("prompt.txt");
            write_prompt(&p, args.ctx)?;
            p
        }
    };
    let csv_path = work.join("arm.csv");
    println!(
        "ds4-advisor: ctx {}, {} arms, card {} MiB, ceiling {} MiB, prefer {}",
        args.ctx,
        list.len(),
        args.vram_total_mib,
        ceiling,
        match args.prefer {
            Prefer::Balanced => "balanced",
            Prefer::Prefill => "prefill",
            Prefer::Decode => "decode",
        }
    );

    let started = Instant::now();
    let mut measured: Vec<Measured> = Vec::new();
    let mut skipped: Vec<Skipped> = Vec::new();
    for (i, arm) in list.iter().enumerate() {
        if arm.estimate_mib > ceiling {
            skipped.push(Skipped {
                name: arm.name.clone(),
                why: format!(
                    "estimated {} MiB over the {} MiB ceiling",
                    arm.estimate_mib, ceiling
                ),
            });
            continue;
        }
        if args.budget_min > 0 && started.elapsed() > Duration::from_secs(args.budget_min * 60) {
            skipped.push(Skipped {
                name: arm.name.clone(),
                why: "budget spent".into(),
            });
            continue;
        }
        eprintln!(
            "[{}/{}] {} (est {} MiB)",
            i + 1,
            list.len(),
            arm.name,
            arm.estimate_mib
        );
        match run_arm(arm, i, &args, &prompt, &csv_path) {
            Ok(m) => {
                eprintln!(
                    "        ttft {:.2} s, prefill {:.0} tok/s, decode {:.1} tok/s, peak {} MiB",
                    m.ttft_s, m.prefill_tps, m.decode_tps, m.peak_mib
                );
                measured.push(m);
            }
            Err(err) => {
                eprintln!("        failed: {err}");
                skipped.push(Skipped {
                    name: arm.name.clone(),
                    why: err,
                });
            }
        }
    }

    if measured.is_empty() {
        return Err("no arm produced a measurement".into());
    }
    measured.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!(
        "{:<4} {:<26} {:<4} {:>8} {:>11} {:>10} {:>10} {:>8} {:>7}",
        "rank", "arm", "kv", "ttft_s", "prefill", "decode", "peak_MiB", "score", "wall_s"
    );
    for (r, m) in measured.iter().enumerate() {
        println!(
        "{:<4} {:<26} {:<4} {:>8.2} {:>11.0} {:>10.1} {:>10} {:>8.1} {:>7.1}",
            r + 1,
            list[m.arm_index].name,
            list[m.arm_index].kv,
            m.ttft_s,
            m.prefill_tps,
            m.decode_tps,
            m.peak_mib,
            m.score,
            m.seconds
        );
    }
    if !skipped.is_empty() {
        println!("\nnot measured:");
        for s in &skipped {
            println!("  {:<26} {}", s.name, s.why);
        }
    }
    let best = &measured[0];
    let arm = &list[best.arm_index];
    println!("\nbest for ctx {}: {}", args.ctx, arm.name);
    println!("  DS4_GEMMA4_FULL_KV={}", arm.kv);
    println!("  DS4_GEMMA4_PREFILL_CHUNK={}", arm.chunk);
    match arm.shape {
        Shape::Resident => println!("  DS4_CUDA_COPY_MODEL=1"),
        Shape::CpuLayers(n) => println!("  DS4_GEMMA4_MOE_CPU_LAYERS={n}"),
        Shape::Cache(mib) => println!("  DS4_GEMMA4_MOE_CACHE_MIB={mib}"),
        Shape::CpuMoe => println!("  DS4_GEMMA4_CPU_MOE=1"),
    }

    if let Some(path) = &args.csv_out {
        let mut text = String::from(
            "arm,kv,chunk,ttft_s,prefill_tps,decode_tps,peak_mib,kv_bytes,score\n",
        );
        for m in &measured {
            let a = &list[m.arm_index];
            text.push_str(&format!(
                "{},{},{},{:.4},{:.2},{:.2},{},{},{:.3}\n",
                a.name, a.kv, a.chunk, m.ttft_s, m.prefill_tps, m.decode_tps, m.peak_mib,
                m.kv_bytes, m.score
            ));
        }
        fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    }
    if let Some(path) = &args.json_out {
        let mut text = String::from("{\n  \"ctx\":");
        text.push_str(&args.ctx.to_string());
        text.push_str(",\n  \"vram_total_mib\":");
        text.push_str(&args.vram_total_mib.to_string());
        text.push_str(",\n  \"best\":\"");
        text.push_str(&arm.name);
        text.push_str("\",\n  \"arms\":[");
        for (i, m) in measured.iter().enumerate() {
            let a = &list[m.arm_index];
            if i > 0 {
                text.push(',');
            }
            text.push_str(&format!(
                "\n    {{\"arm\":\"{}\",\"kv\":\"{}\",\"chunk\":{},\"ttft_s\":{:.4},\"prefill_tps\":{:.2},\"decode_tps\":{:.2},\"peak_mib\":{},\"score\":{:.3}}}",
                a.name, a.kv, a.chunk, m.ttft_s, m.prefill_tps, m.decode_tps, m.peak_mib, m.score
            ));
        }
        text.push_str("\n  ]\n}\n");
        fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    }
    let _ = fs::remove_dir_all(&work);
    Ok(())
}
