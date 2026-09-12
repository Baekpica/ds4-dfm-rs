use crate::{
    cli,
    compare::{self, Comparison, Verdict},
    experiment::{self, Scout, Workload},
    fit::Fit,
    knobs,
    ncu::Counters,
    report, runner,
};
use ds4_perf::artifact::{self, Artifact, Payload, Reference};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub name: String,
    pub environment: BTreeMap<String, String>,
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    candidates: Vec<Candidate>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trial {
    pub candidate: Candidate,
    pub accepted: bool,
    pub comparisons: Vec<Reference>,
    pub verdicts: Vec<Verdict>,
    pub failure: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub automatic: bool,
    pub candidates: Vec<Candidate>,
    pub trials: Vec<Trial>,
    pub selected: Option<Candidate>,
    pub stop_reason: String,
    pub evidence: Vec<String>,
    pub method: String,
}

impl Payload for Decision {
    const KIND: &'static str = "decision";
    fn validate(&self) -> Result<(), String> {
        if self.method != "bracketed-controls-v1" || self.stop_reason.is_empty() {
            return Err("invalid optimization decision".into());
        }
        let mut accepted = BTreeMap::new();
        for trial in &self.trials {
            if trial.accepted
                && (trial.failure.is_some()
                    || trial.comparisons.len() != 2
                    || trial.verdicts.len() != 2
                    || trial.verdicts.iter().any(|v| *v != Verdict::Improved))
            {
                return Err("accepted trial lacks bracketed comparison".into());
            }
            if !trial.accepted && trial.failure.as_ref().is_none_or(|v| v.is_empty()) {
                return Err("rejected trial lacks a recorded reason".into());
            }
            if trial.accepted {
                accepted.extend(trial.candidate.environment.clone());
            }
        }
        match &self.selected {
            Some(selected) if selected.environment != accepted || accepted.is_empty() => {
                return Err("selected controls lack accepted trial evidence".into())
            }
            None if !accepted.is_empty() => {
                return Err("accepted controls missing from selection".into())
            }
            _ => {}
        }
        Ok(())
    }
}

fn candidates(
    scout: &Scout,
    fit: Option<&Fit>,
    counters: Option<&Counters>,
) -> (Vec<Candidate>, Vec<String>) {
    let Some(workload) = &scout.workload else {
        return (
            Vec::new(),
            vec!["Provide --workload before choosing runtime experiments".into()],
        );
    };
    let family = workload.family.to_ascii_lowercase();
    let mut reasons = Vec::new();
    let mut host = false;
    let mut wide = false;
    let mut prefill = false;
    for (name, phase) in &scout.evidence.phases {
        let diagnosis = report::classify(phase);
        reasons.push(format!(
            "{name}: {diagnosis:?}; wall_ns={:?}, busy_ns={:?}",
            phase.wall_ns, phase.busy_ns
        ));
        // Initial runtime proposals target prefill. Decode still contributes
        // diagnostic context and must pass the later regression comparison.
        if name == "ds4.prefill" {
            prefill |= diagnosis != report::Diagnosis::Unknown;
            host |= diagnosis == report::Diagnosis::HostIdle;
            wide |= diagnosis == report::Diagnosis::Fragmentation;
        }
    }
    if let Some(fit) = fit {
        if let Some(bound) = fit.bounds.iter().find(|b| b.launch.phase == "ds4.prefill") {
            reasons.push(format!(
                "top prefill geometry: {} grid={:?} block={:?} waves_lower={} tail_fill={:.3}",
                bound.launch.kernel,
                bound.launch.grid,
                bound.launch.block,
                bound.waves_lower,
                bound.last_wave_fill
            ));
            wide |= bound.waves_lower == 1 && bound.last_wave_fill < 0.5;
        }
        reasons.push(format!(
            "measured envelope: copy_GB_s={:?}, fp32_GFLOP_s={:?}, launch_us={:?}",
            fit.observed_copy_gb_s, fit.observed_fp32_gflop_s, fit.observed_launch_us
        ));
    }
    if let Some(counters) = counters {
        for target in &counters.targets {
            for metric in &target.metrics {
                if metric.name == "sm__warps_active.avg.pct_of_peak_sustained_active" {
                    reasons.push(format!(
                        "{} observed active warps: {:?} {}",
                        target.phase, metric.value, metric.unit
                    ));
                    if target.phase == "ds4.prefill" {
                        wide |= metric.value.is_some_and(|v| v < 30.0);
                    }
                }
            }
        }
    }
    let mut result = Vec::new();
    if family.starts_with("qwen") && host {
        let current = scout
            .environment
            .get("DS4_QWEN_PLE_WORKERS")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(32);
        for workers in [(current * 2).min(64), (current / 2).max(1)] {
            if workers == current {
                continue;
            }
            result.push(Candidate {name:format!("ple-workers-{workers}"),environment:BTreeMap::from([("DS4_QWEN_PLE_WORKERS".into(),workers.to_string())]),reason:"Host GPU-free spans: test PLE worker parallelism with fixed cache allocation; not a claim that PLE caused the gap".into()});
        }
    }
    if !prefill {
        reasons.push("No measured prefill phase for automatic prefill controls".into());
        return (result, reasons);
    }
    let control = if family.starts_with("qwen") {
        Some(("DS4_QWEN_PREFILL_CHUNK", 256, 16384))
    } else if family.starts_with("dots") {
        Some(("DS4_DOTS3_PREFILL_CHUNK", 4096, 8192))
    } else if family.starts_with("solar")
        && scout
            .environment
            .get("DS4_CUDA_SOLAR_GQA_GROUPED")
            .is_none_or(|v| v != "0")
    {
        Some(("DS4_CUDA_SOLAR_GQA_CHUNK", 64, 2048))
    } else {
        None
    };
    if let Some((key, default, maximum)) = control {
        let frontier = scout
            .samples
            .first()
            .and_then(|s| s.rows.iter().map(|r| r.ctx).max())
            .unwrap_or(maximum) as u32;
        let limit = (maximum as u32).min(frontier).max(1);
        let current = scout
            .environment
            .get(key)
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(default)
            .min(limit);
        let values = if wide {
            [
                current.saturating_mul(2),
                current / 2,
                current.saturating_mul(4),
            ]
        } else {
            [current / 2, current.saturating_mul(2), current / 4]
        };
        for value in values.map(|v| v.clamp(1, limit)) {
            if value == current
                || knobs::validate(key, &value.to_string(), &family).is_err()
                || result
                    .iter()
                    .any(|c| c.environment.get(key) == Some(&value.to_string()))
            {
                continue;
            }
            result.push(Candidate {name:format!("chunk-{value}"),environment:BTreeMap::from([(key.into(),value.to_string())]),reason:if wide {"Fragmentation/underfilled geometry: test work granularity before changing kernels"} else {"Test work decomposition at a smaller/larger chunk while preserving model, context and cache policy"}.into()});
        }
    }
    (result, reasons)
}

fn validate_plan(plan: &[Candidate], workload: &Workload) -> Result<(), String> {
    if plan.is_empty() || plan.len() > 32 {
        return Err("experiment plan needs 1..32 candidates".into());
    }
    let mut names = std::collections::BTreeSet::new();
    for candidate in plan {
        if candidate.name.is_empty()
            || !names.insert(&candidate.name)
            || candidate.reason.is_empty()
            || candidate.environment.len() != 1
        {
            return Err(
                "each candidate needs a unique name, reason and one controlled variable".into(),
            );
        }
        for (key, value) in &candidate.environment {
            knobs::validate(key, value, &workload.family)?;
        }
    }
    Ok(())
}

fn replay(
    source: &Scout,
    out: PathBuf,
    workload: &Path,
    repeats: u32,
    overrides: &BTreeMap<String, String>,
    source_root: &Path,
) -> Result<cli::Scout, String> {
    let mut environment = source.environment.clone();
    environment.extend(overrides.clone());
    #[cfg(unix)]
    let command = {
        use std::os::unix::ffi::OsStringExt;
        source
            .command
            .iter()
            .cloned()
            .map(std::ffi::OsString::from_vec)
            .collect()
    };
    #[cfg(not(unix))]
    let command = source
        .command
        .iter()
        .map(|v| {
            String::from_utf8(v.clone())
                .map(Into::into)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    for input in [
        &source.replay.helper,
        &source.replay.collector,
        &source.replay.sdk,
    ]
    .into_iter()
    .flatten()
    {
        if artifact::hash(&input.path)? != input.sha256 {
            return Err("profiling helper/collector binary changed since source scout".into());
        }
    }
    if source.collector == "nsys" {
        let collector = source
            .replay
            .collector
            .as_ref()
            .ok_or("source scout lacks nsys executable identity")?;
        if runner::resolve(std::ffi::OsStr::new("nsys")).as_ref() != Some(&collector.path) {
            return Err("nsys executable resolution changed since source scout".into());
        }
    }
    Ok(cli::Scout {
        exact_environment: true,
        budget: source.budget.clone(),
        out,
        collector: if source.collector == "cupti" {
            cli::Collector::Cupti
        } else {
            cli::Collector::Nsys
        },
        // A phase-only source can still run an explicit controlled experiment.
        fit: source.replay.calibration.is_some(),
        ncu: false,
        proof: true,
        machine: None,
        calibration: source
            .replay
            .calibration
            .as_ref()
            .map(|r| source_root.join(&r.path)),
        gpu_helper: source.replay.helper.as_ref().map(|v| v.path.clone()),
        device: source
            .gpu
            .as_ref()
            .ok_or("automatic experiments need verified CUDA properties")?
            .ordinal,
        cupti_library: source
            .replay
            .collector
            .as_ref()
            .filter(|_| source.collector == "cupti")
            .map(|v| v.path.clone()),
        cupti_sdk: source.replay.sdk.as_ref().map(|v| v.path.clone()),
        repeats,
        workload: Some(workload.to_path_buf()),
        cache_policy: source.cache_policy.clone(),
        environment: environment
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect(),
        command,
    })
}

fn compare_trial(base: &Path, candidate: &Path, out: &Path) -> Result<Verdict, String> {
    compare::run(&cli::Compare {
        baseline: base.join("scout.json"),
        candidate: candidate.join("scout.json"),
        out: out.into(),
        regression: false,
        max_slowdown_percent: 3.0,
        logit_atol: 0.0001,
        logit_rtol: 0.0001,
        logit_rel_rms: 0.0,
    })?;
    let artifact: Artifact<Comparison> = artifact::load(&out.join("compare.json"))?;
    let data = artifact.data.ok_or("comparison has no result")?;
    Ok(data.verdict)
}

fn automatic(args: &cli::Optimize, source: &Scout, decision: &mut Decision) -> Result<(), String> {
    if !source.process_scope.verified {
        return Err("automatic execution needs matching GPU process snapshots".into());
    }
    if !source.unreviewed_environment.is_empty() {
        return Err("automatic replay requires reviewed DS4/CUDA/LD environment controls".into());
    }
    let workload = source
        .workload
        .as_ref()
        .ok_or("--auto requires hashed --workload evidence")?;
    if source.cache_policy != "warmup-then-fresh" {
        return Err("--auto requires an explicit cache policy on the source scout".into());
    }
    if std::env::current_dir().map_err(|e| e.to_string())? != source.cwd {
        return Err("run --auto from the source scout working directory".into());
    }
    if artifact::hash(&source.executable)? != source.executable_sha256 {
        return Err("source benchmark executable changed".into());
    }
    workload.verify()?;
    validate_plan(&decision.candidates, workload)?;
    let workload_path = args
        .out
        .join("workload.json")
        .canonicalize()
        .unwrap_or_else(|_| args.out.join("workload.json"));
    fs::write(
        &workload_path,
        serde_json::to_vec_pretty(workload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let workload_path = workload_path.canonicalize().map_err(|e| e.to_string())?;
    let source_root = args.scout.parent().unwrap_or(Path::new("."));
    let execute = |out: &Path, env: &BTreeMap<String, String>| -> Result<(), String> {
        let remaining = args
            .budget
            .limits()
            .bytes
            .saturating_sub(crate::process::bytes(&args.out)?)
            / (1024 * 1024);
        if remaining == 0 {
            return Err("automatic campaign output budget reached".into());
        }
        let mut replay = replay(
            source,
            out.to_path_buf(),
            &workload_path,
            args.repeats,
            env,
            source_root,
        )?;
        replay.budget = args.budget.clone();
        replay.budget.max_output_mib = remaining;
        runner::scout(&replay)?;
        crate::process::check_bytes(&args.out, args.budget.limits())
    };
    let mut control_env = BTreeMap::new();
    let mut before = args.out.join("control-00");
    execute(&before, &control_env)?;
    for (index, candidate) in decision
        .candidates
        .iter()
        .take(args.rounds as usize)
        .enumerate()
    {
        let candidate_dir = args.out.join(format!("candidate-{:02}", index + 1));
        let mut env = control_env.clone();
        env.extend(candidate.environment.clone());
        let work = execute(&candidate_dir, &env);
        let after = args.out.join(format!("control-{:02}", index + 1));
        // Re-measure the unchanged control after the candidate to expose drift.
        let control = execute(&after, &control_env);
        let mut trial = Trial {
            candidate: candidate.clone(),
            accepted: false,
            comparisons: Vec::new(),
            verdicts: Vec::new(),
            failure: work
                .as_ref()
                .err()
                .cloned()
                .or_else(|| control.as_ref().err().cloned()),
        };
        if work.is_ok() && control.is_ok() {
            for (label, base) in [("before", &before), ("after", &after)] {
                let comparison = args.out.join(format!("compare-{:02}-{label}", index + 1));
                match compare_trial(base, &candidate_dir, &comparison) {
                    Ok(verdict) => trial.verdicts.push(verdict),
                    Err(error) => {
                        trial.failure = Some(format!("{label} comparison: {error}"));
                    }
                }
                let path = comparison.join("compare.json");
                if path.is_file() {
                    match artifact::reference(&path, &args.out) {
                        Ok(reference) => trial.comparisons.push(reference),
                        Err(error) => {
                            trial.failure = Some(format!("{label} reference: {error}"));
                        }
                    }
                }
            }
            trial.accepted = trial.failure.is_none()
                && trial.comparisons.len() == 2
                && trial.verdicts.len() == 2
                && trial.verdicts.iter().all(|v| *v == Verdict::Improved);
            if !trial.accepted && trial.failure.is_none() {
                trial.failure = Some(format!("candidate not retained: {:?}", trial.verdicts));
            }
        }
        if trial.accepted {
            control_env = env;
            decision.selected = Some(Candidate {name:candidate.name.clone(),environment:control_env.clone(),reason:"Improved against both bracketing controls with full-vocabulary and greedy-token parity".into()});
            before = candidate_dir;
        } else {
            before = after;
        }
        decision.trials.push(trial);
        if control.is_err() {
            return Err("control run failed; optimization stopped".into());
        }
    }
    decision.stop_reason = if decision.candidates.len() > args.rounds as usize {
        "round budget reached"
    } else {
        "candidate plan exhausted"
    }
    .into();
    Ok(())
}

pub fn run(args: &cli::Optimize) -> Result<(), String> {
    artifact::directory(&args.out)?;
    let mut decision = Decision {
        automatic: args.automatic,
        candidates: Vec::new(),
        trials: Vec::new(),
        selected: None,
        stop_reason: "analysis complete".into(),
        evidence: Vec::new(),
        method: "bracketed-controls-v1".into(),
    };
    let mut refs = Vec::new();
    let result = (|| {
        let source = experiment::load(&args.scout)?;
        refs.push(artifact::reference(&args.scout, &args.out)?);
        let root = args.scout.parent().unwrap_or(Path::new("."));
        let fit = root.join("fit.json");
        let fit: Option<Artifact<Fit>> = if fit.exists() {
            refs.push(artifact::reference(&fit, &args.out)?);
            Some(artifact::load(&fit)?)
        } else {
            None
        };
        let ncu = root.join("ncu.json");
        let ncu: Option<Artifact<Counters>> = if ncu.exists() {
            refs.push(artifact::reference(&ncu, &args.out)?);
            Some(artifact::load(&ncu)?)
        } else {
            None
        };
        let scout = source.require()?;
        let (suggested, reasons) = candidates(
            scout,
            fit.as_ref()
                .filter(|a| a.complete)
                .and_then(|a| a.data.as_ref()),
            ncu.as_ref()
                .filter(|a| a.complete)
                .and_then(|a| a.data.as_ref()),
        );
        decision.candidates = suggested;
        decision.evidence = reasons;
        if let Some(path) = &args.plan {
            let plan: Plan = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            validate_plan(
                &plan.candidates,
                scout
                    .workload
                    .as_ref()
                    .ok_or("--plan requires source workload identity")?,
            )?;
            decision.candidates = plan.candidates;
            refs.push(artifact::reference(path, &args.out)?);
        }
        if args.automatic {
            automatic(args, scout, &mut decision)?;
        }
        crate::process::check_bytes(&args.out, args.budget.limits())
    })();
    let mut result = result;
    let mut artifact = Artifact::new(decision);
    artifact.complete = result.is_ok();
    if let Err(e) = &result {
        artifact.data.as_mut().unwrap().stop_reason = e.clone();
        artifact.warnings.push(e.clone());
    }
    refs.extend(experiment::raw_refs(&args.out, &args.out)?);
    artifact.inputs = refs;
    let final_bytes = serde_json::to_vec_pretty(&artifact)
        .map_err(|e| e.to_string())?
        .len() as u64
        + 1;
    if result.is_ok()
        && crate::process::bytes(&args.out)?.saturating_add(final_bytes)
            > args.budget.limits().bytes
    {
        let error = "automatic campaign output budget exceeded at publication".to_string();
        artifact.complete = false;
        artifact.data.as_mut().unwrap().stop_reason = error.clone();
        artifact.warnings.push(error.clone());
        result = Err(error);
    }
    artifact::save(&args.out.join("decision.json"), &artifact)?;
    println!("decision: {}", args.out.join("decision.json").display());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_uncontrolled_plan() {
        let workload = Workload {
            protocol: "ds4-bench-v1".into(),
            name: "test".into(),
            family: "qwen".into(),
            files: BTreeMap::new(),
            shape: BTreeMap::new(),
            cache_state: "warm".into(),
        };
        let candidate = Candidate {
            name: "bad".into(),
            reason: "test".into(),
            environment: BTreeMap::from([("DS4_QWEN_PLE_CACHE_MB".into(), "512".into())]),
        };
        assert!(validate_plan(&[candidate], &workload).is_err());
    }
}
