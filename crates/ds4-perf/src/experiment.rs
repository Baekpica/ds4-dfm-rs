use crate::{bench, cli, csv, inspect, nsys, runner};
use ds4_perf::{
    artifact::{self, Artifact, Payload, Reference},
    machine::{Gpu, Machine},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputFile {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub protocol: String,
    pub name: String,
    pub family: String,
    pub files: BTreeMap<String, InputFile>,
    pub shape: BTreeMap<String, u64>,
    pub cache_state: String,
}

impl Workload {
    pub fn verify(&self) -> Result<(), String> {
        if self.name.is_empty()
            || self.family.is_empty()
            || self.cache_state.is_empty()
            || !self.files.contains_key("model")
            || !self.files.contains_key("prompt")
        {
            return Err(
                "workload requires name, family, cache_state and hashed model/prompt files".into(),
            );
        }
        for (name, input) in &self.files {
            if artifact::hash(&input.path)? != input.sha256 {
                return Err(format!("workload input changed: {name}"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub rows: Vec<bench::Row>,
    pub proof_files: BTreeMap<String, Reference>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scout {
    pub process_scope: ProcessScope,
    pub unreviewed_environment: Vec<String>,
    pub model_argument: Option<PathBuf>,
    pub replay: Replay,
    pub budget: cli::Budget,
    pub requested_repeats: u32,
    pub requested_proof: bool,
    pub collector: String,
    pub gpu: Option<Gpu>,
    pub command: Vec<Vec<u8>>,
    pub cwd: PathBuf,
    pub executable: PathBuf,
    pub executable_sha256: String,
    pub environment: BTreeMap<String, String>,
    pub workload: Option<Workload>,
    pub cache_policy: String,
    pub samples: Vec<Sample>,
    pub evidence: nsys::Evidence,
    pub timing_source: String,
}

impl Payload for Scout {
    const KIND: &'static str = "scout";
    fn validate(&self) -> Result<(), String> {
        if self.process_scope.verified
            && (self.process_scope.before != self.process_scope.expected_owners
                || self.process_scope.after != self.process_scope.expected_owners)
        {
            return Err("GPU process snapshot verification disagrees with recorded PIDs".into());
        }
        if let Some(gpu) = &self.gpu {
            gpu.validate()?;
        }
        if !["nsys", "cupti"].contains(&self.collector.as_str())
            || self.command.is_empty()
            || self.executable_sha256.len() != 64
            || self.timing_source != "fresh-unprofiled-processes"
        {
            return Err("invalid scout identity or timing source".into());
        }
        for sample in &self.samples {
            for row in &sample.rows {
                if row.ctx == 0
                    || row.prefill_tokens == 0
                    || row.gen_tokens == 0
                    || [row.prefill_tps, row.gen_tps, row.first_token_sec]
                        .into_iter()
                        .any(|v| !v.is_finite() || v <= 0.0)
                {
                    return Err("invalid benchmark sample".into());
                }
            }
        }
        for p in self.evidence.phases.values() {
            if [p.wall_ns, p.projected_ns, p.busy_ns, p.mem_ns, p.gap_ns]
                .into_iter()
                .flatten()
                .any(|v| !v.is_finite() || v < 0.0)
                || p.kernels
                    .iter()
                    .any(|k| !k.total_ns.is_finite() || k.total_ns < 0.0)
            {
                return Err("invalid phase evidence".into());
            }
        }
        Ok(())
    }
}

pub struct Prepared {
    collector: Option<InputFile>,
    limits: crate::process::Limits,
    calibration: Option<PathBuf>,
    pub machine: Artifact<Machine>,
    pub environment: BTreeMap<OsString, OsString>,
    workload: Option<Workload>,
    executable: PathBuf,
    executable_sha256: String,
}

impl Prepared {
    pub fn run(&self, command: &[OsString], out: &Path, name: &str) -> Result<(), String> {
        let mut command = command.to_vec();
        if command.first().is_some_and(|v| v == "nsys") {
            command[0] = self
                .collector
                .as_ref()
                .ok_or("nsys executable unavailable")?
                .path
                .clone()
                .into();
        }
        runner::run_limited(&command, out, name, &self.environment, self.limits)
    }
    pub fn pin(&self, args: &cli::Scout) -> cli::Scout {
        let mut args = args.clone();
        args.command[0] = self.executable.clone().into_os_string();
        args.calibration = self.calibration.clone();
        args
    }
}

pub fn prepare(out: &Path, args: &cli::Scout) -> Result<Prepared, String> {
    let environment = if args.exact_environment {
        runner::replay_controls(&args.environment)?
    } else {
        runner::controls(&args.environment)?
    };
    runner::device_environment(&environment)?;
    let workload: Option<Workload> = args
        .workload
        .as_ref()
        .map(|p| {
            let bytes = fs::read(p).map_err(|e| e.to_string())?;
            let mut workload: Workload =
                serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            for input in workload.files.values_mut() {
                input.path = p
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(&input.path)
                    .canonicalize()
                    .map_err(|e| e.to_string())?;
            }
            workload.verify_scope(&args.command, &environment)?;
            workload.verify()?;
            Ok::<_, String>(workload)
        })
        .transpose()?;
    let executable = runner::resolve(&args.command[0]).ok_or("benchmark executable not found")?;
    let executable_sha256 = artifact::hash(&executable)?;
    artifact::directory(&out.join("machine"))?;
    let machine = inspect::collect(
        &out.join("machine"),
        args.device,
        args.gpu_helper.as_deref(),
        None,
    )?;
    if let Some(expected) = &args.machine {
        let previous: Artifact<Machine> = artifact::load(expected)?;
        let a = previous
            .require()?
            .gpu
            .as_ref()
            .ok_or("input machine has no GPU")?;
        let b = machine
            .require()?
            .gpu
            .as_ref()
            .ok_or("current machine has no GPU")?;
        if !a.same_device(b) {
            return Err("selected machine differs from current CUDA device/driver".into());
        }
        artifact::bundle::<Machine>(expected, &out.join("machine-input"))?;
    }
    if args.proof
        && args
            .command
            .iter()
            .any(|a| a == "--dump-frontier-logits-dir")
    {
        return Err("--proof owns --dump-frontier-logits-dir; remove the benchmark flag".into());
    }
    let calibration = args
        .calibration
        .as_ref()
        .map(|path| {
            artifact::bundle::<ds4_perf::machine::Calibration>(path, &out.join("calibration"))
        })
        .transpose()?;
    let collector = if matches!(args.collector, cli::Collector::Nsys) {
        runner::resolve(std::ffi::OsStr::new("nsys"))
            .map(|path| {
                Ok::<_, String>(InputFile {
                    sha256: artifact::hash(&path)?,
                    path,
                })
            })
            .transpose()?
    } else {
        None
    };
    if collector.is_some() {
        fs::write(
            out.join("collector-config.json"),
            serde_json::to_vec_pretty(&Replay {
                helper: None,
                collector: collector.clone(),
                sdk: None,
                calibration: None,
            })
            .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(Prepared {
        collector,
        calibration,
        limits: args.budget.limits(),
        machine,
        environment,
        workload,
        executable,
        executable_sha256,
    })
}

pub fn bench_command(args: &cli::Scout, out: &Path, name: &str) -> Vec<OsString> {
    let mut command = args.command.clone();
    if args.proof {
        command.push("--dump-frontier-logits-dir".into());
        command.push(out.join(format!("{name}-proof")).into_os_string());
    }
    command
}

pub fn sample(out: &Path, name: &str) -> Result<Sample, String> {
    let rows =
        bench::parse(csv::open(&out.join(format!("{name}.stdout"))).map_err(|e| e.to_string())?)?;
    let mut proof_files = BTreeMap::new();
    let proof = out.join(format!("{name}-proof"));
    if proof.is_dir() {
        for entry in fs::read_dir(&proof).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            if entry.file_type().map_err(|e| e.to_string())?.is_file() {
                proof_files.insert(
                    entry.file_name().to_string_lossy().into(),
                    artifact::reference(&entry.path(), out)?,
                );
            }
        }
    }
    Ok(Sample { rows, proof_files })
}

// Every local raw file is referenced directly so consumers can verify a moved
// evidence directory without trusting only an intermediate manifest hash.
pub fn raw_refs(root: &Path, dir: &Path) -> Result<Vec<Reference>, String> {
    let mut paths = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    paths.sort_by_key(|e| e.file_name());
    let mut refs = Vec::new();
    for entry in paths {
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        if kind.is_dir() {
            refs.extend(raw_refs(root, &entry.path())?);
        }
        if kind.is_file() {
            refs.push(artifact::reference(&entry.path(), root)?);
        }
    }
    Ok(refs)
}

pub fn finish(
    out: &Path,
    args: &cli::Scout,
    prepared: Prepared,
    result: &Result<nsys::Evidence, String>,
) -> Result<(), String> {
    let mut samples = Vec::new();
    let mut warnings = Vec::new();
    for index in 0..args.repeats {
        let name = if index == 0 {
            "bench".into()
        } else {
            format!("bench-{index:02}")
        };
        match sample(out, &name) {
            Ok(s) => samples.push(s),
            Err(e) => warnings.push(e),
        }
    }
    let verified = (|| {
        if artifact::hash(&prepared.executable)? != prepared.executable_sha256 {
            return Err("benchmark binary changed during scout".into());
        }
        if let Some(workload) = &prepared.workload {
            workload.verify()?;
        }
        if let Some(input) = &prepared.collector {
            if artifact::hash(&input.path)? != input.sha256 {
                return Err("nsys executable changed during scout".into());
            }
        }
        crate::process::check_bytes(out, args.budget.limits())?;
        Ok::<_, String>(())
    })();
    if let Err(e) = &verified {
        warnings.push(e.clone());
    }
    if let Err(e) = result {
        warnings.push(e.clone());
    }
    #[cfg(unix)]
    let command = {
        use std::os::unix::ffi::OsStrExt;
        args.command.iter().map(|a| a.as_bytes().to_vec()).collect()
    };
    #[cfg(not(unix))]
    let command = args
        .command
        .iter()
        .map(|a| a.to_string_lossy().as_bytes().to_vec())
        .collect();
    let mut replay = Replay {
        helper: None,
        collector: None,
        sdk: None,
        calibration: None,
    };
    if let Some(machine) = &prepared.machine.data {
        if let Some(path) = machine.facts.get("gpu_helper_path") {
            replay.helper = Some(InputFile {
                path: path.into(),
                sha256: machine
                    .facts
                    .get("gpu_helper_sha256")
                    .cloned()
                    .ok_or("helper hash missing")?,
            });
        }
    }
    let collector = out.join("collector-config.json");
    if collector.exists() {
        let collected: Replay =
            serde_json::from_slice(&fs::read(collector).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        replay.collector = collected.collector;
        replay.sdk = collected.sdk;
    }
    replay.calibration = prepared
        .calibration
        .as_ref()
        .map(|path| artifact::reference(path, out))
        .transpose()?;
    let mut scout = Artifact::new(Scout {
        process_scope: process_scope(out, &prepared.environment),
        unreviewed_environment: runner::unreviewed_env(),
        model_argument: crate::workload::model_argument(&args.command).ok(),
        replay,
        budget: args.budget.clone(),
        requested_repeats: args.repeats,
        requested_proof: args.proof,
        collector: match args.collector {
            cli::Collector::Nsys => "nsys",
            cli::Collector::Cupti => "cupti",
        }
        .into(),
        gpu: prepared.machine.data.as_ref().and_then(|m| m.gpu.clone()),
        command,
        cwd: std::env::current_dir().map_err(|e| e.to_string())?,
        executable: prepared.executable,
        executable_sha256: prepared.executable_sha256,
        environment: prepared
            .environment
            .iter()
            .map(|(k, v)| (k.to_string_lossy().into(), v.to_string_lossy().into()))
            .collect(),
        workload: prepared.workload,
        cache_policy: args.cache_policy.clone(),
        samples,
        evidence: result.as_ref().cloned().unwrap_or_default(),
        timing_source: "fresh-unprofiled-processes".into(),
    });
    let data = scout.data.as_ref().unwrap();
    let samples_complete = data.samples.len() == args.repeats as usize
        && data.samples.iter().all(|s| !s.rows.is_empty());
    if !samples_complete {
        warnings.push("requested benchmark samples are missing or empty".into());
    }
    let proof = if args.proof {
        crate::compare::validate_proofs(out, data)
    } else {
        Ok(())
    };
    if let Err(e) = &proof {
        warnings.push(e.clone());
    }
    let scope_ok = data.process_scope.verified;
    if !scope_ok {
        warnings.push(
            "GPU/host process snapshots are unavailable or differ from intended owners".into(),
        );
    }
    scout.complete =
        result.is_ok() && verified.is_ok() && samples_complete && proof.is_ok() && scope_ok;
    scout.warnings = warnings;
    scout.inputs = raw_refs(out, out)?;
    // Publish immutable collection evidence first; final scout includes all
    // requested analysis and its output accounting, without a reference cycle.
    artifact::save(&out.join("collection.json"), &scout)?;
    let mut advanced: Result<(), String> = (|| {
        verified?;
        proof?;
        if !scope_ok {
            return Err("unverified process snapshots; see scout.json".into());
        }
        if !samples_complete && args.repeats > 1 {
            return Err("incomplete repeated benchmark samples; see scout.json".into());
        }
        if result.is_err() {
            return Ok(());
        }
        if args.fit {
            let shape = scout
                .data
                .as_ref()
                .and_then(|s| s.workload.as_ref())
                .map(|w| w.shape.clone())
                .unwrap_or_default();
            crate::fit::run(out, &prepared.machine, args.calibration.as_deref(), shape)?;
        }
        if args.ncu {
            crate::ncu::run(
                out,
                &args.command,
                &prepared.environment,
                &scout,
                args.budget.limits(),
            )?;
        }
        Ok(())
    })();
    scout.inputs = raw_refs(out, out)?;
    // Reserve the terminal status and the final JSON before claiming complete.
    let final_bytes = serde_json::to_vec_pretty(&scout)
        .map_err(|e| e.to_string())?
        .len() as u64
        + 1
        + b"COMPLETE\n".len() as u64;
    if advanced.is_ok()
        && crate::process::bytes(out)?.saturating_add(final_bytes) > args.budget.limits().bytes
    {
        advanced = Err("profiler output budget exceeded at publication".into());
    }
    if let Err(error) = &advanced {
        scout.complete = false;
        scout.warnings.push(error.clone());
    }
    artifact::save(&out.join("scout.json"), &scout)?;
    advanced
}

pub fn failed(out: &Path, message: String) -> Result<(), String> {
    let mut scout = Artifact::<Scout>::failed(message);
    scout.inputs = raw_refs(out, out)?;
    artifact::save(&out.join("scout.json"), &scout)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replay {
    pub helper: Option<InputFile>,
    pub collector: Option<InputFile>,
    pub sdk: Option<InputFile>,
    pub calibration: Option<Reference>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessScope {
    pub expected_owners: Vec<u32>,
    pub before: Vec<u32>,
    pub after: Vec<u32>,
    pub verified: bool,
}

fn process_scope(out: &Path, env: &BTreeMap<OsString, OsString>) -> ProcessScope {
    let mut expected = Vec::new();
    let manifest = env.get(std::ffi::OsStr::new("DS4_CUDA_WEIGHT_IPC_MANIFEST"));
    if let Some(path) = manifest {
        if let Ok(text) = fs::read_to_string(path) {
            expected = text
                .lines()
                .filter_map(|line| {
                    line.strip_prefix("owner ")
                        .and_then(|v| v.split_whitespace().next())
                        .and_then(|v| v.parse().ok())
                })
                .collect();
        }
    }
    expected.sort_unstable();
    let read = |label| -> Result<Vec<u32>, String> {
        let text = fs::read_to_string(out.join(format!("processes-{label}.txt")))
            .map_err(|e| e.to_string())?;
        let text = text
            .strip_prefix("gpu_query_ok=true\n")
            .ok_or("GPU process query failed")?;
        let (text, ps) = text
            .split_once("ps_ok=")
            .ok_or("process snapshot missing")?;
        if !ps.starts_with("true\n") {
            return Err("host process query failed".into());
        }
        let mut pids = Vec::new();
        csv::records(text.as_bytes(), |row| {
            if row.iter().all(|v| v.trim().is_empty()) {
                return Ok(());
            }
            pids.push(
                row[0]
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| "unrecognized GPU process record")?,
            );
            Ok(())
        })?;
        pids.sort_unstable();
        Ok(pids)
    };
    let before = read("before");
    let after = read("after");
    let verified = before.as_ref().is_ok_and(|p| *p == expected)
        && after.as_ref().is_ok_and(|p| *p == expected)
        && (manifest.is_none() || !expected.is_empty());
    ProcessScope {
        expected_owners: expected,
        before: before.unwrap_or_default(),
        after: after.unwrap_or_default(),
        verified,
    }
}
