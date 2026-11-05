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
const EXPERT_RECORDS: u64 = 3_840; // 30 layers x 128 experts, ~1.87 MiB each
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
    /// The static estimate, kept as the offline fallback: the card and the
    /// artifact are not read, so it answers in microseconds and never opens
    /// the model. `--estimate quote` replaces it with the engine's own
    /// serving quote; `auto` tries the quote and falls back to this.
    static_mib: u64,
}

/// Where an arm's memory estimate comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Estimate {
    /// The engine's --check-config quote when the server binary can answer,
    /// the static model otherwise.
    Auto,
    /// The static model only: no model open, no card read beyond the total.
    Static,
    /// The engine's quote only; an arm whose quote cannot be produced is
    /// reported as skipped rather than estimated.
    Quote,
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
    estimate: Estimate,
    dry_run: bool,
    include_expensive: bool,
    /// Add pool arms sized from the card, the requested context and the free
    /// margin: the limited-VRAM setup (128K on a 12 GiB card, 1 GiB free).
    pool_budget: bool,
    /// Run exactly this arm (a name from --dry-run), for one expensive point.
    only: Option<String>,
    /// VRAM free right now (total minus used). The pool budget is priced from
    /// this, not from the card total: a desktop compositor and the driver
    /// already hold memory that the engine can never spend.
    available_mib: u64,
}

#[derive(Clone, Copy)]
struct Measured {
    arm_index: usize,
    /// Client-visible time to the first token: the prefill plus the first
    /// decode step (the bench's own first_token_sec starts after the prefill).
    ttft_s: f64,
    prefill_tps: f64,
    decode_tps: f64,
    peak_mib: u64,
    /// The lowest `memory.free` sampled during the arm: the number the
    /// operator's "keep N MiB free" promise is measured against.
    free_mib: u64,
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
        "  --ctx N            context to size the sweep for (required, max 262144)",
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
        "  --estimate MODE    static (default: offline model), quote (the",
        "                     engine quote only), auto (quote, static fallback)",
        "  --free-margin MiB  VRAM that must stay free at peak (default 384)",
        "  --pool-budget      add the pool arms that fill the card up to that margin",
        "                     (exact KV + workspace + the measured CUDA runtime);",
        "                     the lane uploads every routed expert before use",
        "  --only NAME        run one arm by name (see --dry-run), nothing else",
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
        estimate: Estimate::Static,
        dry_run: false,
        include_expensive: false,
        pool_budget: false,
        only: None,
        available_mib: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{} needs a value", arg));
        match arg.as_str() {
            "--ctx" => args.ctx = value()?.parse().map_err(|_| "--ctx wants a number")?,
            "--gen-tokens" => {
                args.gen_tokens = value()?
                    .parse()
                    .map_err(|_| "--gen-tokens wants a number")?
            }
            "--prefer" => {
                args.prefer = match value()?.as_str() {
                    "balanced" => Prefer::Balanced,
                    "prefill" | "ttft" => Prefer::Prefill,
                    "decode" => Prefer::Decode,
                    other => {
                        return Err(format!(
                            "--prefer wants balanced, prefill or decode (got {other})"
                        ))
                    }
                }
            }
            "--prompt-file" => args.prompt = Some(PathBuf::from(value()?)),
            "--model" => args.model = PathBuf::from(value()?),
            "--bench" => args.bench = PathBuf::from(value()?),
            "--repo" => args.repo = PathBuf::from(value()?),
            "--csv" => args.csv_out = Some(PathBuf::from(value()?)),
            "--json" => args.json_out = Some(PathBuf::from(value()?)),
            "--arm-timeout" => {
                args.arm_timeout = value()?
                    .parse()
                    .map_err(|_| "--arm-timeout wants seconds")?
            }
            "--budget-min" => {
                args.budget_min = value()?.parse().map_err(|_| "--budget-min wants minutes")?
            }
            "--estimate" => {
                args.estimate = match value()?.as_str() {
                    "auto" => Estimate::Auto,
                    "static" => Estimate::Static,
                    "quote" => Estimate::Quote,
                    other => {
                        return Err(format!(
                            "--estimate wants auto, static or quote (got '{other}')"
                        ))
                    }
                };
            }
            "--vram-total" => {
                args.vram_total_mib = value()?.parse().map_err(|_| "--vram-total wants MiB")?
            }
            "--free-margin" => {
                args.free_headroom_mib = value()?.parse().map_err(|_| "--free-margin wants MiB")?
            }
            "--pool-budget" => args.pool_budget = true,
            "--only" => args.only = Some(value()?),
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
    // 65,536 was the resident lane's ceiling on a 12 GiB card (the whole
    // 9.37 GiB map plus the graph); the pool lane keeps the experts
    // host-cold and lifts it. The model itself stops at G4_CONTEXT.
    if args.ctx > 262_144 {
        return Err(format!(
            "--ctx {} is past the model's 262,144-token limit",
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

/// The KV rows the engine actually allocates (ds4_gemma4.h
/// ds4_gemma4_kv_bytes): the five full layers keep the whole context at
/// 2,056 B/token in E4M3 (4,112 bf16), and the 25 sliding layers keep a ring
/// of window + chunk rows at 8,192 B each, so they stop growing with ctx.
/// `kv_mib` above prices only the full layers; at 128K the rings add ~0.3 GiB
/// and a budget that ignored them would over-fill the pool.
fn kv_exact_mib(ctx: u32, kv: &str, chunk: u32) -> u64 {
    const WINDOW: u64 = 1_024;
    let rows = u64::from(chunk.max(7));
    let sliding = 25 * (WINDOW + rows).min(u64::from(ctx)) * 8_192;
    let per_token = if kv == "fp8" {
        KV_FP8_BYTES_PER_TOKEN
    } else {
        KV_BF16_BYTES_PER_TOKEN
    };
    ((sliding + per_token * u64::from(ctx)) / (1024 * 1024)).max(1)
}

/// The session graph's workspace at `chunk` rows, from the engine's own
/// arithmetic (ds4_core::gemma4_workspace_bytes, the same function the
/// serving quote and the fit gate consume).
fn workspace_mib(chunk: u32) -> u64 {
    ds4_core::gemma4_workspace_bytes(chunk) / (1024 * 1024)
}

/// The CUDA runtime the process holds before any weights are promoted:
/// measured 2026-11-05 at 9,083 MiB used with 8,550 MiB of pool and spans
/// resident (2.35 GiB spans + 6.00 GiB pool), so 533 MiB of context, module
/// loads and library workspaces. It is inside `memory.used` but in none of
/// the engine's price lists, which is why the plan-time quote cannot refuse
/// an arm that the boot gate then refuses.
const CUDA_RUNTIME_MIB: u64 = 533;

/// The pool a limited-VRAM machine can hold at `ctx`, `kv` and `chunk` while
/// keeping `free_margin` MiB free at peak, priced from the VRAM free now
/// rather than the card total: the compositor and the driver already hold
/// pages the engine can never spend (measured 249 MiB on the 4070, which is
/// why a total-priced arm at 128K finished with 820 MiB free instead of the
/// 1,024 the budget promised).
fn pool_budget_mib(free_mib: u64, ctx: u32, kv: &str, chunk: u32, free_margin: u64) -> u64 {
    let fixed = RESIDENT_SET_MIB
        + kv_exact_mib(ctx, kv, chunk)
        + workspace_mib(chunk)
        + CUDA_RUNTIME_MIB
        + free_margin;
    free_mib.saturating_sub(fixed).min(EXPERT_MIB)
}

fn estimate_mib(ctx: u32, kv: &str, shape: Shape, chunk: u32) -> u64 {
    let kv = kv_exact_mib(ctx, kv, chunk);
    match shape {
        // The resident lane copies the whole map and runs one shared graph.
        Shape::Resident => MODEL_MIB + kv + workspace_mib(chunk) + CUDA_RUNTIME_MIB,
        Shape::CpuLayers(n) => {
            let n = n.min(LAYERS as u32) as u64;
            RESIDENT_SET_MIB
                + EXPERT_MIB * (LAYERS - n) / LAYERS
                + kv
                + workspace_mib(chunk)
                + CUDA_RUNTIME_MIB
        }
        // The host-cold pool lane keeps only the promoted spans resident.
        Shape::Cache(mib) => {
            RESIDENT_SET_MIB + u64::from(mib) + kv + workspace_mib(chunk) + CUDA_RUNTIME_MIB
        }
        Shape::CpuMoe => RESIDENT_SET_MIB + kv + workspace_mib(chunk) + CUDA_RUNTIME_MIB,
    }
}

/// The engine's own serving quote for one arm, in MiB: the server binary is
/// asked for --check-config with exactly the environment the arm runs with,
/// and `quote.total` is the number its admission check uses. This is the
/// portable path - the same code answers on a discrete card and on GB10's
/// coherent memory - so `--estimate auto` prefers it and the static model is
/// only the fallback. None means the binary is absent or would not answer.
/// Does this bench binary know the knob? Its table is compiled in, so a
/// binary built before the knob landed accepts the flag name at the usage
/// level and then ignores it - which is exactly how the cache lane stayed
/// unreachable under ds4-bench (PHASE C0): the binary predated the placement
/// knobs, --gemma4-moe-cache-mib was rejected as unsupported, and from_env()
/// carried nothing either. Asking for --help with the knob first is the
/// cheapest probe that separates the two.
fn bench_knows(bench: &Path, flag: &str) -> bool {
    match Command::new(bench)
        .arg(flag)
        .arg("1")
        .arg("--help")
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(out) => !String::from_utf8_lossy(&out.stderr).contains("unsupported"),
        Err(_) => false,
    }
}

fn quote_total_mib(args: &Args, arm: &Arm) -> Option<u64> {
    let server = args.repo.join("ds4-server");
    if !server.is_file() {
        return None;
    }
    let mut child = Command::new(&server)
        .arg("-m")
        .arg(&args.model)
        .arg("--cuda")
        .arg("-c")
        .arg(args.ctx.to_string())
        .arg("--mem-floor-gb")
        .arg("0")
        .arg("--max-seqs")
        .arg("1")
        .arg("--check-config")
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", std::env::var("HOME").unwrap_or_default())
        .envs(passthrough_env())
        .envs(arm_env(arm, args))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    /* An engine open takes tens of seconds and must never hang the sweep. */
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(200));
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.trim_start().starts_with('{'))?;
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let total = value.get("quote")?.get("total")?.as_u64()?;
    Some(total.div_ceil(1024 * 1024))
}

/// The estimate an arm is judged by, and the source it came from so the
/// report can say which one answered.
fn estimate_for(args: &Args, arm: &Arm) -> (u64, &'static str) {
    match args.estimate {
        Estimate::Static => (arm.static_mib, "static"),
        Estimate::Quote => match quote_total_mib(args, arm) {
            Some(mib) => (mib, "quote"),
            None => (0, "quote-unavailable"),
        },
        Estimate::Auto => match quote_total_mib(args, arm) {
            Some(mib) => (mib, "quote"),
            None => (arm.static_mib, "static"),
        },
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
            static_mib: estimate_mib(args.ctx, kv, shape, chunk),
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
    if args.pool_budget {
        // The limited-VRAM shape: fill the card up to the free margin at the
        // requested context. Every routed expert is uploaded into a slot
        // before the kernel reads it (MOE_MISS=evict, pinned in arm_env), so
        // the GPU always computes the active routes; the pool decides how
        // often a route costs a fill, not whether it runs on the GPU.
        let budget = pool_budget_mib(
            args.available_mib,
            args.ctx,
            "fp8",
            512,
            args.free_headroom_mib,
        );
        for (name, mib) in [
            ("pool-budget-fp8", budget),
            ("pool-two-thirds-fp8", budget * 2 / 3),
            ("pool-half-fp8", budget / 2),
        ] {
            // A pool below 1 GiB cannot hold even a layer's union and the
            // engine's floor is G4_USED experts; below that the arm is noise.
            if mib >= 1024 {
                push(name, "fp8", Shape::Cache(mib as u32), 512);
            }
        }
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
        // The GPU is the only expert compute unit: a slot miss fills the
        // record into a slot and the GPU runs the route, never the host.
        ("DS4_GEMMA4_MOE_MISS", "evict".into()),
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

/// The card's own free figure. It is smaller than total-minus-used by the
/// reserved pages the driver never hands out (measured 387 MiB on the 4070),
/// so it is the number a "keep N MiB free" promise has to be checked against.
fn nvidia_free_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
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
    // The ratio is measured, not assumed: the technical-prose fixture runs at
    // 2.9 bytes per token, while this repeated paragraph measured 4.8 on the
    // first live run (1,290 tokens from 6,144 bytes), which made every arm fail
    // with "prompt has 1290 tokens, need at least N". Seven bytes per token
    // leaves margin over the worse of the two, and an undersized prompt still
    // fails loudly instead of being silently benchmarked below the frontier.
    const PARA: &str = "A dense vector quantized model keeps its experts in a compact \
codebook and rebuilds every weight row from an index, a gain and a residual plane, so the \
decoder reads a small number of bytes for a large number of weights. This paragraph exists \
only to give the prefill stage realistic prose to chew on, with ordinary sentence lengths and \
the usual punctuation, because the frontier, not the wording, is what the sweep measures.\n";
    let want_bytes = (ctx as usize).saturating_mul(7);
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
    let mut low_free = u64::MAX;
    let mut timed_out = false;
    loop {
        if let Some(used) = nvidia_used_mib() {
            peak = peak.max(used);
        }
        if let Some(free) = nvidia_free_mib() {
            low_free = low_free.min(free);
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
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| "the arm wrote an empty CSV".to_string())?
        .split(',')
        .collect();
    // Resolve by name: the first version read fixed offsets and silently took
    // gen_tokens as prefill throughput, so every arm reported 32 tok/s.
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| format!("CSV has no {name} column"))
    };
    let (c_ctx, c_pf_tokens, c_pf_tps, c_gen_tps, c_ttft, c_kv) = (
        col("ctx_tokens")?,
        col("prefill_tokens")?,
        col("prefill_tps")?,
        col("gen_tps")?,
        col("first_token_sec")?,
        col("kvcache_bytes")?,
    );
    let row = lines
        .last()
        .ok_or_else(|| "the arm wrote no data row".to_string())?;
    let f: Vec<&str> = row.split(',').collect();
    let at = |i: usize| f.get(i).copied().unwrap_or("");
    let prefill_tokens = at(c_pf_tokens).parse::<f64>().unwrap_or(0.0);
    let prefill_tps = at(c_pf_tps).parse::<f64>().unwrap_or(0.0);
    let decode_tps = at(c_gen_tps).parse::<f64>().unwrap_or(0.0);
    // Time to the first token, as a client waits for it: the bench's
    // first_token_sec starts counting after the prefill, so at 128K it read
    // 0.04 s while the request had been running for three minutes. Prefill
    // dominates this number at long context, which is exactly what the
    // ranking should see.
    let decode_start_s = at(c_ttft).parse::<f64>().unwrap_or(f64::INFINITY);
    let ttft_s = if prefill_tps > 0.0 {
        decode_start_s + prefill_tokens / prefill_tps
    } else {
        decode_start_s
    };
    let kv_bytes = at(c_kv).parse::<u64>().unwrap_or(0);
    let frontier: u32 = at(c_ctx).parse().unwrap_or(0);
    if !(prefill_tps > 0.0) || !(decode_tps > 0.0) || !(ttft_s > 0.0) {
        return Err(format!(
            "the arm reported no usable timings (frontier {frontier}): {row}"
        ));
    }
    // An arm is only a measurement of the lane it names: a host-mapped run
    // with no map installed peaks near 1.2 GiB, and the cache lane through
    // ds4-bench does exactly that today (reproduced: DS4_GEMMA4_MOE_CACHE_MIB
    // = 2048, peak 1,217 MiB, decode 6.5 tok/s). Reporting those numbers as
    // "cache4g" would be a wrong answer with a confident face.
    let floor_mib = match arm.shape {
        Shape::Resident => 8_000,
        Shape::CpuLayers(_) | Shape::Cache(_) | Shape::CpuMoe => 2_200,
    };
    if peak < floor_mib {
        return Err(format!(
            "the lane did not engage: peak {peak} MiB is below the {floor_mib} MiB floor"
        ));
    }
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
        // The card's own free reading when it answered, else the total minus
        // the sampled peak (which ignores the driver's reserved pages).
        free_mib: if low_free == u64::MAX {
            args.vram_total_mib.saturating_sub(peak)
        } else {
            low_free
        },
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

/// VRAM free on the card right now, from the card's own memory.free. A served
/// endpoint, a compositor or another job, and the driver's reserved pages,
/// are already gone from this number.
fn free_now_mib(args: &Args) -> u64 {
    nvidia_free_mib().unwrap_or(args.vram_total_mib)
}

/// The arm list for this invocation: the full sweep, or the one arm `--only`
/// names. Also the up-front check that the bench binary knows the cache
/// knobs: an old binary accepts the flag name and carries nothing (PHASE C0).
fn build_list(args: &Args) -> Result<Vec<Arm>, String> {
    let list = match &args.only {
        // One expensive point (a 128K arm is minutes): run exactly that arm,
        // with the same environment, threshold and report as the sweep.
        Some(only) => {
            let filtered: Vec<Arm> = arms(args)
                .into_iter()
                .filter(|arm| arm.name == *only)
                .collect();
            if filtered.is_empty() {
                return Err(format!(
                    "no arm named '{only}'; --dry-run lists the names at this ctx"
                ));
            }
            filtered
        }
        None => arms(args),
    };
    if list.iter().any(|a| matches!(a.shape, Shape::Cache(_))) {
        for flag in ["--gemma4-moe-cache-mib", "--gemma4-moe-staging-mib"] {
            if !bench_knows(&args.bench, flag) {
                return Err(format!(
                    "{} does not know {flag} (built before the knob landed); \
rebuild it: make ds4-bench",
                    args.bench.display()
                ));
            }
        }
    }
    Ok(list)
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

    if args.dry_run {
        // A dry run never touches the server, so the pool budget it prints is
        // priced from whatever the card has free right now.
        args.available_mib = free_now_mib(&args);
        let list = build_list(&args)?;
        println!(
            "card {} MiB, free now {} MiB, ceiling {} MiB ({} MiB must stay free), ctx {}",
            args.vram_total_mib, args.available_mib, ceiling, args.free_headroom_mib, args.ctx
        );
        for (i, arm) in list.iter().enumerate() {
            let (est, source) = estimate_for(&args, arm);
            let fit = if source == "quote-unavailable" {
                "the engine quote did not answer"
            } else if est <= ceiling {
                "fits"
            } else {
                "estimated over the ceiling"
            };
            println!(
                "{:>2}. {:<26} kv {:<4} chunk {:<5} est {:>5} MiB ({})  {}",
                i + 1,
                arm.name,
                arm.kv,
                arm.chunk,
                est,
                source,
                fit
            );
        }
        if args.pool_budget {
            println!(
                "\nThe pool budget is priced from the {} MiB free right now; stop a served \
                 endpoint (./run-gemma4.sh serve stop) to price it from the idle card.",
                args.available_mib
            );
        }
        println!("\nNote: the estimates only prune; nothing is measured until a real run.");
        return Ok(());
    }

    // Stop the endpoint before anything is priced: with it up, the pool
    // budget would be priced against a card it is already holding, and a
    // --pool-budget --only NAME request would not even find its arm.
    let _guard = EndpointGuard::new(&args.repo, args.stop_endpoint, args.port)?;
    args.available_mib = free_now_mib(&args);
    let list = build_list(&args)?;
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
        let (estimate, source) = estimate_for(&args, arm);
        if source == "quote-unavailable" {
            skipped.push(Skipped {
                name: arm.name.clone(),
                why: "the engine quote did not answer (--estimate quote)".into(),
            });
            continue;
        }
        if estimate > ceiling {
            skipped.push(Skipped {
                name: arm.name.clone(),
                why: format!(
                    "estimated {} MiB ({}) over the {} MiB ceiling",
                    estimate, source, ceiling
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
            "[{}/{}] {} (est {} MiB, {})",
            i + 1,
            list.len(),
            arm.name,
            estimate,
            source
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
    // The free margin is a measured property, not an estimate: an arm whose
    // sampled peak leaves less than --free-margin is not a candidate for a
    // limited-VRAM machine, whatever its throughput. (The static/quote
    // estimate only prunes; the card decides.)
    let mut over: Vec<Measured> = Vec::new();
    let fitting: Vec<Measured> = measured
        .drain(..)
        .filter(|m| {
            let fits = m.free_mib >= args.free_headroom_mib;
            if !fits {
                over.push(*m);
            }
            fits
        })
        .collect();
    measured = fitting;
    for m in &over {
        // For a pool arm the correction is arithmetic: the pool was as much
        // too wide as the sampled free memory missed the margin.
        let deficit = args.free_headroom_mib.saturating_sub(m.free_mib);
        let hint = match list[m.arm_index].shape {
            Shape::Cache(mib) => format!(
                "; DS4_GEMMA4_MOE_CACHE_MIB={} would meet it",
                u64::from(mib).saturating_sub(deficit)
            ),
            _ => String::new(),
        };
        skipped.push(Skipped {
            name: list[m.arm_index].name.clone(),
            why: format!(
                "peak {} MiB, lowest free {} MiB, under the {} MiB margin{}",
                m.peak_mib, m.free_mib, args.free_headroom_mib, hint
            ),
        });
    }
    if measured.is_empty() {
        return Err(format!(
            "no arm kept the {} MiB margin (lowest free: {})",
            args.free_headroom_mib,
            over.iter()
                .map(|m| format!("{} {} MiB", list[m.arm_index].name, m.free_mib))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    measured.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!();
    println!(
        "{:<4} {:<26} {:<4} {:>8} {:>11} {:>10} {:>10} {:>9} {:>8} {:>7}",
        "rank",
        "arm",
        "kv",
        "ttft_s",
        "prefill",
        "decode",
        "peak_MiB",
        "free_MiB",
        "score",
        "wall_s"
    );
    for (r, m) in measured.iter().enumerate() {
        println!(
            "{:<4} {:<26} {:<4} {:>8.2} {:>11.0} {:>10.1} {:>10} {:>9} {:>8.1} {:>7.1}",
            r + 1,
            list[m.arm_index].name,
            list[m.arm_index].kv,
            m.ttft_s,
            m.prefill_tps,
            m.decode_tps,
            m.peak_mib,
            m.free_mib,
            m.score,
            m.seconds
        );
    }
    println!(
        "\nttft_s is what a client waits for: the prefill plus the first decode \
         token. free_MiB is the lowest memory.free sampled during the arm, the \
         number the margin is checked against."
    );
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
        Shape::Cache(mib) => {
            println!("  DS4_GEMMA4_MOE_CACHE_MIB={mib}");
            println!(
                "    {} of {} expert records resident; a miss fills the slot before the \
                 GPU runs the route (DS4_GEMMA4_MOE_MISS=evict)",
                u64::from(mib) * EXPERT_RECORDS / EXPERT_MIB,
                EXPERT_RECORDS
            );
        }
        Shape::CpuMoe => println!("  DS4_GEMMA4_CPU_MOE=1"),
    }
    println!(
        "  DS4_GEMMA4_FULL_KV={} at ctx {} keeps {} MiB free at peak",
        arm.kv, args.ctx, best.free_mib
    );

    if let Some(path) = &args.csv_out {
        let mut text = String::from(
            "arm,kv,chunk,ttft_s,prefill_tps,decode_tps,peak_mib,free_mib,kv_bytes,score\n",
        );
        for m in &measured {
            let a = &list[m.arm_index];
            text.push_str(&format!(
                "{},{},{},{:.4},{:.2},{:.2},{},{},{},{:.3}\n",
                a.name,
                a.kv,
                a.chunk,
                m.ttft_s,
                m.prefill_tps,
                m.decode_tps,
                m.peak_mib,
                m.free_mib,
                m.kv_bytes,
                m.score
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
                "\n    {{\"arm\":\"{}\",\"kv\":\"{}\",\"chunk\":{},\"ttft_s\":{:.4},\"prefill_tps\":{:.2},\"decode_tps\":{:.2},\"peak_mib\":{},\"free_mib\":{},\"score\":{:.3}}}",
                a.name, a.kv, a.chunk, m.ttft_s, m.prefill_tps, m.decode_tps, m.peak_mib,
                m.free_mib, m.score
            ));
        }
        text.push_str("\n  ]\n}\n");
        fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    }
    let _ = fs::remove_dir_all(&work);
    Ok(())
}
