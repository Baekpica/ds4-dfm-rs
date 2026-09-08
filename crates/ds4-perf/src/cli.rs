use clap::{Args, Parser, Subcommand, ValueEnum};
use std::{ffi::OsString, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Reproducible CUDA performance evidence and experiments"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Inspect machine capabilities; optionally measure the device envelope.
    Inspect(Inspect),
    /// Compatibility entry point for the v0.1.0 capability report.
    #[command(hide = true)]
    Doctor {
        #[arg(long, default_value = "./ds4-bench")]
        bench: String,
    },
    /// Profile a fresh benchmark and identify execution-path bottlenecks.
    Scout(Scout),
    /// Compare matched unprofiled experiments and correctness evidence.
    Compare(Compare),
    /// Choose experiments from evidence; --auto executes a bounded loop.
    Optimize(Optimize),
}

#[derive(Args, Debug)]
pub struct Inspect {
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub calibrate: bool,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long)]
    pub gpu_helper: Option<PathBuf>,
    #[arg(long)]
    pub bench: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Collector {
    Nsys,
    Cupti,
}

#[derive(Args, Clone, Debug)]
pub struct Scout {
    #[arg(skip)]
    pub exact_environment: bool,
    #[command(flatten)]
    pub budget: Budget,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, value_enum, default_value = "nsys")]
    pub collector: Collector,
    #[arg(long)]
    pub fit: bool,
    /// Request ds4-bench token and full-vocabulary frontier proof files.
    #[arg(long)]
    pub proof: bool,
    /// CUPTI injection library (default: beside ds4-perf-gpu).
    #[arg(long, requires = "collector")]
    pub cupti_library: Option<PathBuf>,
    /// NVIDIA libcupti.so used for NVTX injection and activity collection.
    #[arg(long)]
    pub cupti_sdk: Option<PathBuf>,
    #[arg(long)]
    pub ncu: bool,
    #[arg(long)]
    pub machine: Option<PathBuf>,
    #[arg(long, requires = "fit")]
    pub calibration: Option<PathBuf>,
    #[arg(long)]
    pub gpu_helper: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=100))]
    pub repeats: u32,
    /// Explicit workload/cache contract used when comparing runs.
    #[arg(long)]
    pub workload: Option<PathBuf>,
    #[arg(long, default_value = "inherited", value_parser = ["inherited", "warmup-then-fresh"])]
    pub cache_policy: String,
    /// Allowlisted runtime control, passed directly to the child process.
    #[arg(long = "env")]
    pub environment: Vec<String>,
    #[arg(last = true, required = true)]
    pub command: Vec<OsString>,
}

#[derive(Args, Debug)]
pub struct Compare {
    #[arg(long)]
    pub baseline: PathBuf,
    #[arg(long)]
    pub candidate: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    /// Exit unsuccessfully on regression or insufficient comparison evidence.
    #[arg(long)]
    pub regression: bool,
    #[arg(long, default_value_t = 3.0)]
    pub max_slowdown_percent: f64,
    #[arg(long, default_value_t = 0.0001)]
    pub logit_atol: f64,
    #[arg(long, default_value_t = 0.0001)]
    pub logit_rtol: f64,
}

#[derive(Args, Debug)]
pub struct Optimize {
    #[command(flatten)]
    pub budget: Budget,
    #[arg(long)]
    pub scout: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    /// Run evidence-selected experiments and compare each candidate.
    #[arg(long = "auto")]
    pub automatic: bool,
    #[arg(long)]
    pub plan: Option<PathBuf>,
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=32))]
    pub rounds: u32,
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(3..=100))]
    pub repeats: u32,
}

#[derive(Args, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    /// Deadline for each benchmark/profiler process, including descendants.
    #[arg(long,default_value_t=1800,value_parser=clap::value_parser!(u64).range(1..=86400))]
    pub timeout_seconds: u64,
    /// Total raw output budget per scout; auto also bounds the campaign.
    #[arg(long,default_value_t=2048,value_parser=clap::value_parser!(u64).range(1..=65536))]
    pub max_output_mib: u64,
}
impl Budget {
    pub fn limits(&self) -> crate::process::Limits {
        crate::process::Limits {
            timeout: std::time::Duration::from_secs(self.timeout_seconds),
            bytes: self.max_output_mib.saturating_mul(1024 * 1024),
        }
    }
}
