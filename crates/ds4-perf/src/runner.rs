use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Clone)]
pub struct Capture {
    pub ok: bool,
    pub out: String,
    pub err: String,
}

impl Capture {
    pub fn failed(error: &str) -> Self {
        Self {
            ok: false,
            out: String::new(),
            err: error.into(),
        }
    }
    pub fn combined(&self) -> String {
        format!("{}\n{}", self.out, self.err)
    }
}

pub trait Probe {
    fn capture(&mut self, program: &str, args: &[&str]) -> Capture;
}

pub struct System;
impl Probe for System {
    fn capture(&mut self, program: &str, args: &[&str]) -> Capture {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let out = std::env::temp_dir().join(format!("ds4-probe-{}-{nonce}", std::process::id()));
        if let Err(error) = fs::create_dir(&out) {
            return Capture::failed(&error.to_string());
        }
        let mut command = vec![OsString::from(program)];
        command.extend(args.iter().map(OsString::from));
        let result = controls(&[]).and_then(|env| {
            run_limited(
                &command,
                &out,
                "probe",
                &env,
                crate::process::Limits {
                    timeout: std::time::Duration::from_secs(60),
                    bytes: 64 * 1024 * 1024,
                },
            )
        });
        let stdout = fs::read(out.join("probe.stdout")).unwrap_or_default();
        let mut stderr = fs::read(out.join("probe.stderr")).unwrap_or_default();
        if let Err(error) = &result {
            stderr.extend_from_slice(format!("\n{error}").as_bytes());
        }
        let _ = fs::remove_dir_all(&out);
        Capture {
            ok: result.is_ok(),
            out: String::from_utf8_lossy(&stdout).into(),
            err: String::from_utf8_lossy(&stderr).into(),
        }
    }
}

pub fn write(path: &Path, text: &str) -> Result<(), String> {
    fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

// Quoting is for reproduction only: execution never invokes a shell.
pub fn shell(command: &[OsString]) -> String {
    command
        .iter()
        .map(|s| quote(&s.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

pub fn run(command: &[OsString], out: &Path, name: &str) -> Result<(), String> {
    run_with(command, out, name, &controls(&[])?)
}

pub fn run_with(
    command: &[OsString],
    out: &Path,
    name: &str,
    env: &std::collections::BTreeMap<OsString, OsString>,
) -> Result<(), String> {
    run_limited(command, out, name, env, crate::process::Limits::default())
}

pub fn run_limited(
    command: &[OsString],
    out: &Path,
    name: &str,
    env: &std::collections::BTreeMap<OsString, OsString>,
    limits: crate::process::Limits,
) -> Result<(), String> {
    if command.is_empty() {
        return Err("empty process command".into());
    }
    write(&out.join(format!("{name}.command.txt")), &shell(command))?;
    let stdout = File::create(out.join(format!("{name}.stdout"))).map_err(|e| e.to_string())?;
    let stderr = File::create(out.join(format!("{name}.stderr"))).map_err(|e| e.to_string())?;
    let start = std::time::Instant::now();
    let mut process = Command::new(&command[0]);
    for key in PERF_ENV {
        process.env_remove(key);
    }
    process
        .args(&command[1..])
        .env_remove("CUDA_INJECTION64_PATH")
        .env_remove("NVTX_INJECTION64_PATH")
        .env_remove("DS4_PERF_CUPTI_OUTPUT")
        .envs(env)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr);
    let status = crate::process::run(&mut process, out, limits);
    let success = status.as_ref().is_ok_and(|s| s.success());
    write(
        &out.join(format!("{name}.status.txt")),
        &format!(
            "status={status:?}\nwall_sec={}\ntimeout_sec={}\noutput_budget_bytes={}\n",
            start.elapsed().as_secs_f64(),
            limits.timeout.as_secs(),
            limits.bytes
        ),
    )?;
    if success {
        Ok(())
    } else {
        Err(format!(
            "{name} failed ({status:?}); see {}/{}.stderr",
            out.display(),
            name
        ))
    }
}

// Explicitly reviewed runtime controls only. Never record arbitrary DS4_* keys.
const PERF_ENV: &[&str] = &[
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "CUDA_CACHE_PATH",
    "CUDA_HOME",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_MAXSIZE",
    "CUDA_FORCE_PTX_JIT",
    "CUDA_LAUNCH_BLOCKING",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "DS4_WEIGHT_SERVER",
    "DS4_CUDA_WEIGHT_IPC_MANIFEST",
    "DS4_CUDA_WEIGHT_IPC_SCOPE",
    "DS4_CUDA_WEIGHT_IPC_NO_DRAFTER",
    "DS4_WEIGHT_RESIDENCY",
    "DS4_WEIGHT_RESIDENCY_BASE",
    "DS4_WEIGHT_RESIDENCY_MTP",
    "DS4_WEIGHT_RESIDENCY_DRAFTER",
    "DS4_CUDA_KEEP_MODEL_PAGES",
    "DS4_CUDA_DIRECT_MODEL",
    "DS4_CUDA_NO_FD_CACHE",
    "DS4_CUDA_NO_HBM_CACHE",
    "DS4_MEMGOV",
    "DS4_MEMGOV_HEADROOM_MB",
    "DS4_SESSION_LAZY_GRAPH",
    "DS4_CUDA_CAPTURE",
    "DS4_CUDA_NO_GRAPH",
    "DS4_CUDA_FP8_KV",
    "DS4_CUDA_FP4_INDEX",
    "DS4_CUDA_MMQ",
    "DS4_MTP_SPEC_DISABLE",
    "DS4_QWEN_BATCH",
    "DS4_QWEN_PREFILL_CHUNK",
    "DS4_QWEN_PLE_CACHE_MB",
    "DS4_QWEN_PLE_WORKERS",
    "DS4_QWEN_PLE_DIR",
    "DS4_PLE_CUDA_TILE_ROWS",
    "DS4_PLE_NO_BATCH_ACQUIRE",
    "DS4_QWEN_NO_SWIGLU_Q8_EMIT",
    "DS4_QWEN_QSA_NO_FUSED",
    "DS4_QWEN_PLE_NO_LOOKAHEAD",
    "DS4_QWEN_MTP_SEQUENTIAL_VERIFY",
    "DS4_SOLAR_KV_FORMAT",
    "DS4_SOLAR_MOE_RESIDUAL",
    "DS4_CUDA_SOLAR_GQA_GROUPED",
    "DS4_CUDA_SOLAR_GQA_CHUNK",
    "DS4_DOTS3_PREFILL_CHUNK",
    "DS4_INKLING_NO_LINEAR",
    "DS4_INKLING_NO_MOE_BATCH",
    "DS4_INKLING_NO_Q8_BATCH",
    "DS4_DOTS3_ATTN_NO_HMMA",
    "DS4_DOTS3_ATTN_NO_SPLIT",
    "DS4_DOTS3_VALUE_NO_HMMA",
    "DS4_DOTS3_VALUE_NO_DECODE",
    "DS4_DOTS3_ABSORB_NO_HMMA",
    "DS4_DOTS3_NO_FUSED",
    "DS4_QWEN_PREFILL_OPENING",
    "DS4_QWEN_HC_NO_FUSED_MIX",
    "DS4_CUDA_NO_NORM_Q8EMIT",
    "DS4_MMQ_DENSE_D2R",
    "DS4_MMQ_D2R_MAX_K",
    "DS4_MMQ_D2R",
    "DS4_MMQ_D2R_IQ2",
    "DS4_MMQ_D2R_MIN_COLS",
    "DS4_MMQ_NO_YIND",
    "DS4_MMQ_PIPE",
    "DS4_MMQ_PIPE_MAX_X",
    "DS4_MMQ_IQ1M_PREFILL",
    "DS4_MMQ_IQ1M_SLOT_LOOP",
    "DS4_MMQ_IQ1M_WORKLIST",
    "DS4_MMQ_IQ1_PAIR",
    "DS4_MMQ_IQ2XS_WORKLIST",
    "DS4_PLE_LATENCY_STATS",
    "DS4_MMQ_WORKLIST",
    "DS4_MMQ_IQ1S_WORKLIST",
    "DS4_MMQ_IQ2XXS_WORKLIST",
    "DS4_SERVER_CONTINUOUS",
    "DS4_CONT_PREFILL_CHUNK",
    "DS4_CONT_CAPTURE",
    "DS4_BATCH_MAX_SEQ",
    "DS4_BATCH_FIT_HEADROOM_MB",
    "DS4_CONT_MTP_MODE",
];

pub fn controls(
    overrides: &[String],
) -> Result<std::collections::BTreeMap<OsString, OsString>, String> {
    let mut env: std::collections::BTreeMap<_, _> = std::env::vars_os()
        .filter(|(k, _)| k.to_str().is_some_and(|k| PERF_ENV.contains(&k)))
        .collect();
    for assignment in overrides {
        let (key, value) = assignment
            .split_once('=')
            .ok_or("--env requires NAME=VALUE")?;
        if !PERF_ENV.contains(&key) {
            return Err(format!("unreviewed performance environment control: {key}"));
        }
        env.insert(key.into(), value.into());
    }
    Ok(env)
}

pub fn environment(vars: impl IntoIterator<Item = (OsString, OsString)>) -> String {
    let mut vars: Vec<_> = vars
        .into_iter()
        .filter(|(key, _)| key.to_str().is_some_and(|k| PERF_ENV.contains(&k)))
        .collect();
    vars.sort();
    let mut out = String::from("# Strict allowlist; all inherited variables still reach both children.\n# This file is not a complete environment or a cache-state proof.\n");
    for (key, value) in vars {
        out.push_str(&format!(
            "{}={}\n",
            key.to_string_lossy(),
            quote(&value.to_string_lossy())
        ));
    }
    out
}

pub fn resolve(program: &OsStr) -> Option<std::path::PathBuf> {
    let path = Path::new(program);
    if path.components().count() > 1 {
        return path.canonicalize().ok();
    }
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|p| p.join(path))
        .find(|p| p.is_file())
        .and_then(|p| p.canonicalize().ok())
}

pub fn scout(args: &crate::cli::Scout) -> Result<(), String> {
    if args.device != 0 {
        return Err("ds4-bench uses visible CUDA device 0; select hardware with CUDA_VISIBLE_DEVICES before running ds4-perf".into());
    }
    if matches!(args.collector, crate::cli::Collector::Nsys)
        && (args.cupti_library.is_some() || args.cupti_sdk.is_some())
    {
        return Err("CUPTI library options require --collector cupti".into());
    }
    let out = &args.out;
    if out.as_os_str().to_string_lossy().contains('%') {
        return Err(
            "output path must not contain Nsight filename substitution character '%'".into(),
        );
    }
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::create_dir(out)
        .map_err(|e| format!("cannot create new output directory {}: {e}", out.display()))?;
    let out = out.canonicalize().map_err(|e| e.to_string())?;
    let result = match crate::experiment::prepare(&out, args) {
        Ok(prepared) => {
            let pinned = prepared.pin(args);
            let work = scout_inner(&out, &pinned, &prepared);
            let finish = crate::experiment::finish(&out, &pinned, prepared, &work);
            work.map(|_| ()).and(finish)
        }
        Err(error) => {
            crate::experiment::failed(&out, error.clone())?;
            Err(error)
        }
    };
    let result = result.and_then(|()| crate::process::check_bytes(&out, args.budget.limits()));
    write(
        &out.join("status.txt"),
        &match &result {
            Ok(()) => "COMPLETE\n".into(),
            Err(err) => format!("FAILED\n{err}\n"),
        },
    )?;
    result
}

fn scout_inner(
    out: &Path,
    args: &crate::cli::Scout,
    prepared: &crate::experiment::Prepared,
) -> Result<crate::nsys::Evidence, String> {
    let command = &args.command;
    use crate::{bench, csv, doctor, nsys, report};
    eprintln!("ds4-perf: capability probes; artifacts {}", out.display());
    let mut system = System;
    let mut d = doctor::inspect(&mut system, None);
    d.facts.insert("benchmark".into(), shell(command));
    d.save_probes(out)?;
    write(&out.join("doctor.txt"), &d.render())?;
    write(&out.join("command.txt"), &format!("{}\n", shell(command)))?;
    // Preserve even non-UTF-8 Unix argv byte-for-byte alongside the shell display.
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let mut argv = Vec::new();
        for arg in command {
            argv.extend_from_slice(arg.as_bytes());
            argv.push(0);
        }
        fs::write(out.join("command.argv"), argv).map_err(|e| e.to_string())?;
    }
    write(
        &out.join("env.txt"),
        &environment(prepared.environment.clone()),
    )?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    let mut manifest = format!(
        "format=ds4-perf-v1\ntimestamp_unix={timestamp}\ncwd={}\nbenchmark_command={}\n",
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .display(),
        shell(command)
    );
    if let Some(binary) = resolve(&command[0]) {
        manifest.push_str(&format!("benchmark_executable={}\n", binary.display()));
        let hash = Command::new("sha256sum").arg(&binary).output();
        if let Ok(hash) = hash {
            if hash.status.success() {
                manifest.push_str(&format!(
                    "benchmark_sha256={}\n",
                    String::from_utf8_lossy(&hash.stdout)
                        .split_whitespace()
                        .next()
                        .unwrap_or("unknown")
                ));
            }
        }
    }
    manifest.push_str("throughput_source=bench.stdout (unprofiled)\nprofiled_throughput=informational only\ncache_policy=inherited; fresh process does not establish cold OS/owner caches\nenvironment_scope=strict allowlist, not a full comparison identity\n");
    save_manifest(out, &d, &manifest)?;
    hygiene(&mut system, out, "before", &mut d.warnings)?;
    let mut evidence = nsys::Evidence::default();
    let mut rows = Vec::new();
    let work = (|| -> Result<(), String> {
        if matches!(args.collector, crate::cli::Collector::Nsys) && !d.caps.can_scout() {
            return Err(
                "nsys CUDA/NVTX collection or required kernel reports unavailable; see doctor.txt"
                    .into(),
            );
        }
        if args.cache_policy == "warmup-then-fresh" {
            prepared.run(command, out, "warmup")?;
        }
        eprintln!("ds4-perf: fresh unprofiled baseline");
        prepared.run(
            &crate::experiment::bench_command(args, out, "bench"),
            out,
            "bench",
        )?;
        let stderr = fs::read_to_string(out.join("bench.stderr")).map_err(|e| e.to_string())?;
        d.benchmark(&stderr, "ds4-bench NVTX: ");
        if d.facts["NVTX"].starts_with("unavailable") {
            return Err("target benchmark lacks NVTX; build with make ds4-bench-perf and pass ./ds4-bench-perf to scout".into());
        }
        match bench::parse(csv::open(&out.join("bench.stdout")).map_err(|e| e.to_string())?) {
            Ok(parsed) => rows = parsed,
            Err(err) => evidence
                .warnings
                .push(format!("baseline CSV unavailable: {err}")),
        }
        if rows.is_empty() {
            evidence.warnings.push(
                "baseline has no recognized CSV; emit ds4-bench CSV on stdout (omit --csv FILE)"
                    .into(),
            );
        }
        write(
            &out.join("baseline.csv"),
            &format!(
                "{}\n{}",
                bench::HEADER,
                rows.iter().map(bench::Row::csv).collect::<String>()
            ),
        )?;
        for index in 1..args.repeats {
            let name = format!("bench-{index:02}");
            if args.cache_policy == "warmup-then-fresh" {
                prepared.run(command, out, &format!("warmup-{index:02}"))?;
            }
            eprintln!("ds4-perf: fresh unprofiled sample {index}");
            prepared.run(
                &crate::experiment::bench_command(args, out, &name),
                out,
                &name,
            )?;
        }
        if args.cache_policy == "warmup-then-fresh" {
            prepared.run(command, out, "profile-warmup")?;
        }
        let warnings = std::mem::take(&mut evidence.warnings);
        evidence = match args.collector {
            crate::cli::Collector::Nsys => {
                eprintln!("ds4-perf: fresh Nsight Systems process");
                prepared.run(
                    &nsys::profile_command(&d.caps, out, command),
                    out,
                    "profile",
                )?;
                if !out.join("trace.nsys-rep").is_file() {
                    return Err(
                        "nsys did not produce trace.nsys-rep; see profile.stdout/stderr".into(),
                    );
                }
                nsys::collect_prepared(&d.caps, out, prepared)
            }
            crate::cli::Collector::Cupti => {
                crate::cupti::capture(out, args, &prepared.environment)?
            }
        };
        evidence.warnings.extend(warnings);
        Ok(())
    })();
    if let Err(err) = &work {
        evidence.warnings.push(format!("INCOMPLETE SCOUT: {err}"));
    }
    hygiene(&mut system, out, "after", &mut d.warnings)?;
    write(&out.join("doctor.txt"), &d.render())?;
    save_manifest(out, &d, &manifest)?;
    normalized(out, &evidence)?;
    let report = report::render(&d, command, &rows, &evidence);
    write(&out.join("report.txt"), &report)?;
    print!("{report}");
    work.map(|_| evidence)
}

fn save_manifest(out: &Path, d: &crate::doctor::Doctor, context: &str) -> Result<(), String> {
    let mut manifest = context.to_string();
    for (name, value) in &d.facts {
        manifest.push_str(&format!("{name}={}\n", value.replace('\n', "\\n")));
    }
    write(&out.join("manifest.txt"), &manifest)
}

fn hygiene(
    probe: &mut impl Probe,
    out: &Path,
    label: &str,
    warnings: &mut Vec<String>,
) -> Result<(), String> {
    let gpu = probe.capture(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
    );
    let processes = probe.capture("ps", &["-eo", "pid=,comm="]);
    write(
        &out.join(format!("processes-{label}.txt")),
        &format!(
            "gpu_query_ok={}\n{}\n{}\nps_ok={}\n{}\n{}",
            gpu.ok, gpu.out, gpu.err, processes.ok, processes.out, processes.err
        ),
    )?;
    if !gpu.ok || !processes.ok {
        warnings.push(format!(
            "process hygiene {label}: could not establish process state; no processes were stopped"
        ));
    } else if !gpu.out.trim().is_empty() {
        warnings.push(format!("process hygiene {label}: resident GPU processes exist (see processes-{label}.txt); check owner/worker contention before comparisons"));
    }
    Ok(())
}

fn normalized(out: &Path, evidence: &crate::nsys::Evidence) -> Result<(), String> {
    let mut phases =
        String::from("phase,range_ns,projected_ns,busy_union_ns,memop_union_ns,largest_gap_ns\n");
    let mut kernels = String::from("phase,kernel,total_ns,instances\n");
    for p in evidence.phases.values() {
        let numbers = [p.wall_ns, p.projected_ns, p.busy_ns, p.mem_ns, p.gap_ns]
            .map(|v| v.map(|v| v.to_string()).unwrap_or_default());
        phases.push_str(&format!("{},{}\n", p.name, numbers.join(",")));
        for k in &p.kernels {
            kernels.push_str(&format!(
                "{},{},{},{}\n",
                p.name,
                crate::csv::field(&k.name),
                k.total_ns,
                k.count.map(|n| n.to_string()).unwrap_or_default()
            ));
        }
    }
    // Empty phase denotes whole-capture CUDA evidence, including setup.
    for k in &evidence.global {
        kernels.push_str(&format!(
            ",{},{},{}\n",
            crate::csv::field(&k.name),
            k.total_ns,
            k.count.map(|n| n.to_string()).unwrap_or_default()
        ));
    }
    write(&out.join("normalized-phases.csv"), &phases)?;
    write(&out.join("normalized-kernels.csv"), &kernels)
}

pub fn replay_controls(
    overrides: &[String],
) -> Result<std::collections::BTreeMap<OsString, OsString>, String> {
    let mut env = controls(overrides)?;
    let keys: std::collections::BTreeSet<_> = overrides
        .iter()
        .filter_map(|v| v.split_once('=').map(|(k, _)| OsString::from(k)))
        .collect();
    env.retain(|k, _| keys.contains(k));
    Ok(env)
}

pub fn device_environment(
    env: &std::collections::BTreeMap<OsString, OsString>,
) -> Result<(), String> {
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    ] {
        if env.get(std::ffi::OsStr::new(key)).cloned() != std::env::var_os(key) {
            return Err(format!("set {key} on ds4-perf itself so inspection and benchmark use the same device/libraries"));
        }
    }
    Ok(())
}

pub fn unreviewed_env() -> Vec<String> {
    std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .filter(|k| {
            ["DS4_", "CUDA_", "LD_"]
                .iter()
                .any(|prefix| k.starts_with(prefix))
                && !PERF_ENV.contains(&k.as_str())
                && ![
                    "CUDA_INJECTION64_PATH",
                    "NVTX_INJECTION64_PATH",
                    "DS4_PERF_CUPTI_OUTPUT",
                ]
                .contains(&k.as_str())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_is_allowlisted() {
        let out = environment(
            [
                ("DS4_QWEN_PLE_CACHE_MB", "2048"),
                ("DS4_QWEN_PLE_DIR", "/models/PLE-FP8"),
                ("DS4_QWEN_PREFILL_OPENING", "1"),
                ("DS4_MMQ_DENSE_D2R", "0"),
                ("DS4_API_KEY", "secret"),
                ("DS4_UNKNOWN", "secret"),
                ("HF_TOKEN", "secret"),
                ("CUDA_VISIBLE_DEVICES", "0"),
            ]
            .map(|(k, v)| (k.into(), v.into())),
        );
        assert!(out.contains("DS4_QWEN_PLE_CACHE_MB='2048'"));
        assert!(out.contains("DS4_QWEN_PLE_DIR='/models/PLE-FP8'"));
        assert!(out.contains("DS4_QWEN_PREFILL_OPENING='1'"));
        assert!(out.contains("DS4_MMQ_DENSE_D2R='0'"));
        assert!(!out.contains("secret"));
        assert_eq!(quote("a'$(x)`id`"), "'a'\\''$(x)`id`'");
    }
}
