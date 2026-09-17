use super::*;
use crate::{csv, experiment::InputFile, runner};
use std::{
    collections::BTreeMap,
    io::BufReader,
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant, SystemTime},
};

const MEMORY_POLL_MS: u64 = 10;
// Sampling evidence tolerates scheduler jitter, but not an unobserved request.
const MEMORY_MAX_GAP_MS: f64 = 100.0;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryPoint {
    elapsed_ms: f64,
    available_bytes: u64,
}

/// Keep samples, not just a claimed minimum, so profile selection can recheck it.
pub(super) fn measure<T: Send>(
    dir: &Path,
    action: impl FnOnce() -> Result<T, String>,
) -> Result<(T, u64), String> {
    let finished = AtomicBool::new(false);
    let start = Instant::now();
    let (result, samples) = std::thread::scope(|scope| {
        let observer = scope.spawn(|| {
            let mut samples = Vec::new();
            loop {
                samples.push(MemoryPoint {
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                    available_bytes: host_available()?,
                });
                if finished.load(Ordering::Acquire) {
                    return Ok::<_, String>(samples);
                }
                std::thread::sleep(Duration::from_millis(MEMORY_POLL_MS));
            }
        });
        let result = action();
        finished.store(true, Ordering::Release);
        (
            result,
            observer.join().map_err(|_| "host memory observer panicked"),
        )
    });
    let samples = samples??;
    let minimum = samples
        .iter()
        .map(|sample| sample.available_bytes)
        .min()
        .ok_or("no host memory samples")?;
    fs::write(
        dir.join("memory.json"),
        serde_json::to_vec_pretty(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok((result?, minimum))
}

pub(super) fn memory_min(dir: &Path, duration_ms: f64) -> Result<u64, String> {
    let samples: Vec<MemoryPoint> =
        serde_json::from_slice(&fs::read(dir.join("memory.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if !duration_ms.is_finite()
        || duration_ms <= 0.0
        || samples.is_empty()
        || samples
            .iter()
            .any(|p| !p.elapsed_ms.is_finite() || p.elapsed_ms < 0.0)
        || samples.windows(2).any(|p| {
            p[1].elapsed_ms <= p[0].elapsed_ms
                || p[1].elapsed_ms - p[0].elapsed_ms > MEMORY_MAX_GAP_MS
        })
        || samples
            .first()
            .is_some_and(|p| p.elapsed_ms > MEMORY_MAX_GAP_MS)
        || samples
            .last()
            .is_some_and(|p| p.elapsed_ms + MEMORY_MAX_GAP_MS < duration_ms)
    {
        return Err("invalid host memory samples".into());
    }
    samples
        .iter()
        .map(|sample| sample.available_bytes)
        .min()
        .ok_or("no host memory samples".into())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ClockRange {
    min: u32,
    max: u32,
}

impl ClockRange {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.min == 0 || self.max < self.min {
            return Err("invalid expected clock range".into());
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Server {
    pid: u32,
    boot_id: String,
    start_ticks: u64,
    executable: String,
    executable_sha256: String,
    argv_sha256: String,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    environment: Option<BTreeMap<String, String>>,
    #[serde(default)]
    unreviewed_environment: Vec<String>,
    #[serde(default)]
    source: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Identity {
    server: Option<Server>,
    inputs: BTreeMap<String, InputFile>,
    expected_clock_range_mhz: Option<ClockRange>,
    limitations: Vec<String>,
}

impl Identity {
    pub(super) fn capture(
        args: &cli::Serving,
        workload: &Workload,
        checked: &mut artifact::Verification,
    ) -> Result<Self, String> {
        let server = args
            .server_pid
            .map(|pid| server(pid, &args.url))
            .transpose()?;
        let mut limitations = Vec::new();
        if server.is_none() {
            limitations.push("server PID/executable identity was not requested".into());
        }
        if workload.inputs.is_empty() {
            limitations.push("model/template/sidecar input hashes were not supplied".into());
        }
        limitations.push("input manifest coverage is operator-declared; loaded model tensors are not introspected".into());
        let identity = Self {
            server,
            inputs: workload.inputs.clone(),
            expected_clock_range_mhz: workload.expected_clock_range_mhz.clone(),
            limitations,
        };
        identity.verify_checked(args, checked)?;
        if let Some(pid) = args.server_pid {
            fs::write(
                args.out.join("server.argv"),
                fs::read(format!("/proc/{pid}/cmdline")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
        }
        fs::write(
            args.out.join("identity.json"),
            serde_json::to_vec_pretty(&identity).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(identity)
    }

    pub(super) fn verify(&self, args: &cli::Serving) -> Result<(), String> {
        let mut checked = artifact::Verification::default();
        self.verify_checked(args, &mut checked)?;
        checked.finish()
    }

    fn verify_checked(
        &self,
        args: &cli::Serving,
        checked: &mut artifact::Verification,
    ) -> Result<(), String> {
        if let Some(expected) = &self.server {
            if &server(expected.pid, &args.url)? != expected {
                return Err("server process/executable changed during serving collection".into());
            }
        }
        for (name, file) in &self.inputs {
            if name.is_empty() {
                return Err(format!("serving input changed: {name}"));
            }
            checked
                .verify(&file.path, &file.sha256)
                .map_err(|error| format!("serving input changed: {name}: {error}"))?;
        }
        Ok(())
    }

    pub(super) fn add_refs(&self, refs: &mut Vec<artifact::Reference>) {
        refs.extend(self.inputs.values().map(|input| artifact::Reference {
            path: input.path.to_string_lossy().into(),
            sha256: input.sha256.clone(),
        }));
    }
}

fn server(pid: u32, origin: &str) -> Result<Server, String> {
    let root = std::path::PathBuf::from(format!("/proc/{pid}"));
    let stat =
        fs::read_to_string(root.join("stat")).map_err(|e| format!("server PID {pid}: {e}"))?;
    let fields = stat
        .rsplit_once(')')
        .ok_or("invalid process stat")?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    let start_ticks = fields
        .get(19)
        .ok_or("missing server start time")?
        .parse::<u64>()
        .map_err(|e| e.to_string())?;
    let port = origin
        .trim_end_matches('/')
        .rsplit_once(':')
        .ok_or("missing server port")?
        .1
        .parse::<u16>()
        .map_err(|e| e.to_string())?;
    let ipv6 = origin.starts_with("http://[::1]:");
    let mut sockets = BTreeSet::new();
    for table in ["net/tcp", "net/tcp6"] {
        let text = fs::read_to_string(root.join(table)).map_err(|e| e.to_string())?;
        for line in text.lines().skip(1) {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.len() < 10 || fields[3] != "0A" {
                continue;
            }
            let address = fields[1].split(':').next().unwrap_or("");
            let matches_address = if ipv6 {
                table == "net/tcp6"
                    && matches!(
                        address,
                        "00000000000000000000000000000000" | "00000000000000000000000001000000"
                    )
            } else {
                table == "net/tcp" && matches!(address, "00000000" | "0100007F")
            };
            if !matches_address {
                continue;
            }
            if fields[1]
                .rsplit_once(':')
                .and_then(|(_, value)| u16::from_str_radix(value, 16).ok())
                == Some(port)
            {
                sockets.insert(format!("socket:[{}]", fields[9]));
            }
        }
    }
    let owns_listener = fs::read_dir(root.join("fd"))
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .filter_map(|entry| fs::read_link(entry.path()).ok())
        .any(|path| sockets.contains(path.to_string_lossy().as_ref()));
    if !owns_listener {
        return Err("server PID does not own the requested listening port".into());
    }
    let exe = root.join("exe");
    let (environment, unreviewed_environment) = server_environment(&root)?;
    Ok(Server {
        pid,
        start_ticks,
        boot_id: fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .map_err(|e| e.to_string())?
            .trim()
            .into(),
        executable: fs::read_link(&exe)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .into(),
        executable_sha256: artifact::hash(&exe)?,
        argv_sha256: artifact::hash(&root.join("cmdline"))?,
        cwd: Some(
            fs::read_link(root.join("cwd"))
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .into(),
        ),
        environment: Some(environment),
        unreviewed_environment,
        source: source_snapshot(&root.join("cwd")).ok(),
    })
}

/// Bind profiles to the actual runtime sources, including new native includes.
pub(super) fn source_snapshot(cwd: &Path) -> Result<Value, String> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(cwd)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("profile requires a source checkout".into());
    }
    let root = std::path::PathBuf::from(
        String::from_utf8(output.stdout)
            .map_err(|e| e.to_string())?
            .trim(),
    );
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(&root)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("cannot enumerate runtime sources".into());
    }
    let mut files = BTreeMap::new();
    for bytes in output.stdout.split(|b| *b == 0).filter(|p| !p.is_empty()) {
        let name = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        let path = Path::new(name);
        if name.starts_with("crates/ds4-perf") || name.starts_with("tests/") {
            continue;
        }
        if !matches!(
            path.extension().and_then(|v| v.to_str()),
            Some(
                "rs" | "c"
                    | "h"
                    | "cu"
                    | "cuh"
                    | "inc"
                    | "cc"
                    | "cpp"
                    | "hpp"
                    | "m"
                    | "mm"
                    | "metal"
                    | "toml"
                    | "lock"
            )
        ) && name != "Makefile"
            && !name.starts_with("crates/")
        {
            continue;
        }
        files.insert(name, artifact::hash(&root.join(path))?);
    }
    if files.is_empty() {
        return Err("no runtime sources in checkout".into());
    }
    Ok(
        serde_json::json!({"root": root, "sha256": artifact::hash_bytes(&serde_json::to_vec(&files).map_err(|e| e.to_string())?)}),
    )
}

fn server_environment(root: &Path) -> Result<(BTreeMap<String, String>, Vec<String>), String> {
    let mut environment = BTreeMap::new();
    let mut unreviewed = Vec::new();
    for assignment in fs::read(root.join("environ"))
        .map_err(|e| e.to_string())?
        .split(|b| *b == 0)
    {
        let Some(split) = assignment.iter().position(|b| *b == b'=') else {
            continue;
        };
        let Ok(name) = std::str::from_utf8(&assignment[..split]) else {
            continue;
        };
        if ![
            "DS4_",
            "CUDA_",
            "NVIDIA_",
            "LD_",
            "CUBLAS_",
            "OMP_",
            "OPENBLAS_",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
        {
            continue;
        }
        if !runner::reviewed_control(name)
            && ![
                "DS4_CONT_PREFILL_CHUNK_LIVE",
                "DS4_CONT_PREFILL_NOFENCE",
                "DS4_DOTS3_BATCH",
                "DS4_EXAONE_PREFILL_CHUNK",
                "DS4_MEM_FLOOR_GB",
                "DS4_MOTIF3_PREFILL_CHUNK",
                "DS4_NATIVE_PREFILL_CHUNK",
                "DS4_SERVER_COALESCE_MAX",
                "DS4_SERVER_FORK",
                "DS4_SERVER_FORK_PARTIAL",
                "DS4_SERVER_PERSIST_MIN_TOKENS",
                "DS4_STEP37_BATCH",
            ]
            .contains(&name)
        {
            unreviewed.push(name.into());
            continue;
        }
        match std::str::from_utf8(&assignment[split + 1..]) {
            Ok(value) => {
                environment.insert(name.into(), value.into());
            }
            Err(_) => unreviewed.push(name.into()),
        }
    }
    unreviewed.sort();
    Ok((environment, unreviewed))
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GpuSample {
    label: String,
    unix_ms: u64,
    device_ordinal: usize,
    uuid: Option<String>,
    name: Option<String>,
    driver: Option<String>,
    sm_clock_mhz: Option<u32>,
    temperature_c: Option<u32>,
    unavailable_reason: Option<String>,
}

impl GpuSample {
    pub(super) fn timestamp_ms(&self) -> u64 {
        self.unix_ms
    }

    pub(super) fn verify_raw(&self, bytes: &[u8]) -> Result<(), String> {
        let mut rows = Vec::new();
        csv::records(BufReader::new(bytes), |row| {
            rows.push(row);
            Ok(())
        })?;
        if rows.len() != 1 || rows[0].len() != 5 {
            return Err("invalid raw GPU telemetry".into());
        }
        let row = &rows[0];
        if self.uuid.as_deref() != Some(row[0].trim())
            || self.name.as_deref() != Some(row[1].trim())
            || self.driver.as_deref() != Some(row[2].trim())
            || self.sm_clock_mhz != row[3].trim().parse().ok()
            || self.temperature_c != row[4].trim().parse().ok()
        {
            return Err("raw GPU telemetry differs from summary".into());
        }
        Ok(())
    }

    pub(super) fn failures(&self, range: Option<&ClockRange>) -> Vec<String> {
        let Some(range) = range else {
            return Vec::new();
        };
        if self.unavailable_reason.is_some() || self.temperature_c.is_none() {
            return vec![format!(
                "{}: incomplete NVIDIA clock/temperature sample",
                self.label
            )];
        }
        match self.sm_clock_mhz {
            Some(clock) if (range.min..=range.max).contains(&clock) => Vec::new(),
            Some(clock) => vec![format!(
                "{}: observed clock {clock} MHz outside expected {}..{} MHz",
                self.label, range.min, range.max
            )],
            None => vec![format!(
                "{}: expected clock contract lacks observed NVIDIA clock",
                self.label
            )],
        }
    }
}

pub(super) fn gpu(args: &cli::Serving, dir: &Path, label: &str) -> Result<GpuSample, String> {
    let mut sample = GpuSample {
        label: label.into(),
        unix_ms: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis() as u64,
        device_ordinal: args.device,
        uuid: None,
        name: None,
        driver: None,
        sm_clock_mhz: None,
        temperature_c: None,
        unavailable_reason: None,
    };
    let result = (|| {
        let executable =
            runner::resolve(std::ffi::OsStr::new("nvidia-smi")).ok_or("nvidia-smi unavailable")?;
        let output = dir.join(format!("{label}.stdout"));
        let mut command = Command::new(executable);
        command
            .arg(format!("--id={}", args.device))
            .args([
                "--query-gpu=uuid,name,driver_version,clocks.current.sm,temperature.gpu",
                "--format=csv,noheader,nounits",
            ])
            .stdout(Stdio::from(
                fs::File::create(&output).map_err(|e| e.to_string())?,
            ))
            .stderr(Stdio::from(
                fs::File::create(dir.join(format!("{label}.stderr"))).map_err(|e| e.to_string())?,
            ));
        let limits = process::Limits {
            timeout: args.budget.limits().timeout.min(Duration::from_secs(10)),
            bytes: args.budget.limits().bytes,
        };
        if !process::run(&mut command, &args.out, limits)?.success() {
            return Err("nvidia-smi query failed".into());
        }
        let mut rows = Vec::new();
        csv::records(
            BufReader::new(fs::File::open(&output).map_err(|e| e.to_string())?),
            |row| {
                rows.push(row);
                Ok(())
            },
        )?;
        if rows.len() != 1 || rows[0].len() != 5 {
            return Err("unexpected NVIDIA query rows".into());
        }
        let row = &rows[0];
        sample.uuid = Some(row[0].trim().into());
        sample.name = Some(row[1].trim().into());
        sample.driver = Some(row[2].trim().into());
        sample.sm_clock_mhz = Some(row[3].trim().parse::<u32>().map_err(|e| e.to_string())?);
        sample.temperature_c = Some(row[4].trim().parse::<u32>().map_err(|e| e.to_string())?);
        Ok::<_, String>(())
    })();
    if let Err(reason) = result {
        if reason == "interrupted" {
            return Err(reason);
        }
        sample.unavailable_reason = Some(reason);
    }
    process::check_bytes(&args.out, args.budget.limits())?;
    fs::write(
        dir.join(format!("{label}.json")),
        serde_json::to_vec_pretty(&sample).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(sample)
}

#[cfg(test)]
mod coverage_tests {
    use super::*;

    #[test]
    fn memory_window_needs_coverage() {
        let dir = std::env::temp_dir().join(format!("ds4-memory-span-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for (times, complete) in [
            (vec![0.0], false),
            (vec![150.0, 200.0], false),
            (vec![0.0, 200.0], false),
            (vec![0.0, 50.0], false),
            (vec![0.0, 50.0, 100.0, 150.0, 205.0], true),
        ] {
            let samples: Vec<_> = times
                .iter()
                .map(|&elapsed_ms| MemoryPoint {
                    elapsed_ms,
                    available_bytes: 1000,
                })
                .collect();
            fs::write(
                dir.join("memory.json"),
                serde_json::to_vec(&samples).unwrap(),
            )
            .unwrap();
            assert_eq!(
                memory_min(&dir, 200.0).is_ok(),
                complete,
                "memory samples: {times:?}"
            );
        }
        let _ = fs::remove_dir_all(dir);
    }
}
