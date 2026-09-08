use crate::{csv, experiment, nsys, runner};
use ds4_perf::artifact::{self, Artifact, Payload};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, ffi::OsString, path::Path};

const METRICS: &[&str] = &[
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__grid_size",
    "launch__block_size",
    "launch__waves_per_multiprocessor",
];

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metric {
    pub name: String,
    pub unit: String,
    pub value: Option<f64>,
    pub unavailable: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub selection: String,
    pub launch: Option<LaunchIdentity>,
    pub phase: String,
    pub kernel: String,
    pub metrics: Vec<Metric>,
    pub captured: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counters {
    pub targets: Vec<Target>,
    pub limits: Vec<String>,
}

impl Payload for Counters {
    const KIND: &'static str = "ncu";
    fn validate(&self) -> Result<(), String> {
        for target in &self.targets {
            if !nsys::PHASES.contains(&target.phase.as_str())
                || (target.captured && target.launch.is_none())
            {
                return Err("invalid NCU phase".into());
            }
            for m in &target.metrics {
                if m.value.is_some_and(|v| !v.is_finite() || v < 0.0)
                    || m.value.is_some() == m.unavailable.is_some()
                {
                    return Err("invalid NCU metric status".into());
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchIdentity {
    pub id: String,
    pub kernel: String,
    pub process: String,
    pub device: String,
    pub grid: String,
    pub block: String,
}

fn identity(input: impl std::io::BufRead, expected: &str) -> Result<LaunchIdentity, String> {
    let mut identity = None;
    csv::table(input, "metric name", |header, row| {
        let get = |name| {
            csv::value(row, header, &[name])
                .map(str::to_string)
                .ok_or_else(|| format!("NCU launch identity lacks {name}"))
        };
        let current = LaunchIdentity {
            id: get("id")?,
            kernel: get("kernel name")?,
            process: get("process id")?,
            device: get("device")?,
            grid: get("grid size")?,
            block: get("block size")?,
        };
        if current.kernel != expected
            || identity
                .as_ref()
                .is_some_and(|previous| *previous != current)
        {
            return Err("NCU launch identity differs from filtered target".into());
        }
        identity = Some(current);
        Ok(())
    })?;
    identity.ok_or("NCU captured no identified launch".into())
}

fn metrics(input: impl std::io::BufRead) -> Result<Vec<Metric>, String> {
    let mut result = BTreeMap::new();
    let mut launches = std::collections::BTreeSet::new();
    csv::table(input, "metric name", |header, row| {
        let name = csv::value(row, header, &["metric name"]).ok_or("NCU metric name missing")?;
        let unit = csv::value(row, header, &["metric unit"]).unwrap_or("");
        let raw = csv::value(row, header, &["metric value"]).unwrap_or("missing");
        let value = csv::number(&raw.replace(',', "")).ok();
        if let Some(id) = csv::value(row, header, &["id"]) {
            launches.insert(id.to_string());
        }
        if result
            .insert(
                name.to_string(),
                Metric {
                    name: name.into(),
                    unit: unit.into(),
                    value,
                    unavailable: value.is_none().then(|| raw.into()),
                },
            )
            .is_some()
        {
            return Err("duplicate NCU metric; expected one bounded launch".into());
        }
        Ok(())
    })?;
    if launches.len() != 1 {
        return Err("NCU needs exactly one launch ID".into());
    }
    for name in METRICS {
        result.entry((*name).into()).or_insert_with(|| Metric {
            name: (*name).into(),
            unit: String::new(),
            value: None,
            unavailable: Some("not reported by installed NCU/device".into()),
        });
    }
    Ok(result.into_values().collect())
}

pub fn run(
    out: &Path,
    command: &[OsString],
    env: &BTreeMap<OsString, OsString>,
    scout: &Artifact<experiment::Scout>,
    limits: crate::process::Limits,
) -> Result<(), String> {
    let dir = out.join("ncu");
    artifact::directory(&dir)?;
    let mut warnings = Vec::new();
    let mut targets = Vec::new();
    let result = (|| {
        runner::run_limited(&["ncu".into(), "--help".into()], &dir, "help", env, limits)?;
        let help = std::fs::read_to_string(dir.join("help.stdout")).map_err(|e| e.to_string())?;
        runner::run_limited(
            &[
                "ncu".into(),
                "--query-metrics".into(),
                "--query-metrics-mode".into(),
                "all".into(),
            ],
            &dir,
            "query",
            env,
            limits,
        )?;
        let query = std::fs::read_to_string(dir.join("query.stdout")).map_err(|e| e.to_string())?;
        let supported: Vec<_> = METRICS
            .iter()
            .filter(|m| query.contains(**m))
            .copied()
            .collect();
        if !supported.contains(&"gpu__time_duration.sum") {
            return Err("NCU cannot measure kernel duration on this device".into());
        }
        for phase in nsys::PHASES {
            let Some(kernel) = scout
                .require()?
                .evidence
                .phases
                .get(phase)
                .and_then(|p| p.kernels.first())
            else {
                warnings.push(format!("{phase}: no measured kernel to target"));
                continue;
            };
            let label = phase.replace('.', "-");
            let mut capture = crate::report::bounded_ncu(&help, phase, &kernel.name, &[])
                .ok_or("NCU lacks bounded NVTX/kernel filtering")?;
            for required in [
                "--target-processes",
                "--print-kernel-base",
                "--rename-kernels",
                "--print-units",
            ] {
                if !help.contains(required) {
                    return Err(format!(
                        "NCU cannot verify bounded process/kernel identity: lacks {required}"
                    ));
                }
            }
            let base = if kernel.name.starts_with("_Z") {
                "mangled"
            } else if kernel.name.contains(['<', '(']) {
                "demangled"
            } else {
                "function"
            };
            capture.extend([
                "--target-processes".into(),
                "application-only".into(),
                "--print-kernel-base".into(),
                base.into(),
                "--rename-kernels".into(),
                "0".into(),
                "--csv".into(),
                "--page".into(),
                "details".into(),
                "--print-units".into(),
                "base".into(),
                "--metrics".into(),
                supported.join(",").into(),
                "--export".into(),
                dir.join(&label).into_os_string(),
            ]);
            capture.extend_from_slice(command);
            let mut captured = runner::run_limited(&capture, &dir, &label, env, limits).is_ok();
            if !dir.join(format!("{label}.ncu-rep")).is_file() {
                captured = false;
            }
            let normalized = metrics(
                csv::open(&dir.join(format!("{label}.stdout"))).map_err(|e| e.to_string())?,
            )?;
            let launch = identity(
                csv::open(&dir.join(format!("{label}.stdout"))).map_err(|e| e.to_string())?,
                &kernel.name,
            );
            if let Err(error) = &launch {
                warnings.push(format!("{phase}: {error}"));
                captured = false;
            }
            captured &= normalized
                .iter()
                .any(|m| m.name == "gpu__time_duration.sum" && m.value.is_some_and(|v| v > 0.0));
            if !captured {
                warnings.push(format!(
                    "{phase}: NCU capture incomplete; see raw stdout/stderr"
                ));
            }
            targets.push(Target {
                selection:"first matching instance of the phase's top aggregate kernel; not necessarily its dominant geometry".into(),
                launch:launch.ok(),
                phase: phase.into(),
                kernel: kernel.name.clone(),
                metrics: normalized,
                captured,
            });
        }
        if targets.is_empty() {
            return Err("NCU requires a measured prefill/decode target".into());
        }
        Ok::<_, String>(())
    })();
    if let Err(e) = &result {
        warnings.push(e.clone());
    }
    let complete = result.is_ok()
        && !targets.is_empty()
        && targets.iter().all(|t| t.captured)
        && warnings.is_empty();
    let mut artifact = Artifact::new(Counters {targets,limits:vec!["One matching launch per phase with application replay and strict all-attribute matching. Replay/cache controls affect counters; NCU throughput is excluded from experiment speed comparisons.".into(),"Missing counters remain unknown. High utilization alone does not establish a causal bottleneck.".into()]});
    artifact.complete = complete;
    artifact.warnings = warnings;
    artifact.inputs = experiment::raw_refs(out, &dir)?;
    for name in ["collection.json", "fit.json"] {
        let path = out.join(name);
        if path.exists() {
            artifact.inputs.push(artifact::reference(&path, out)?);
        }
    }
    artifact::save(&out.join("ncu.json"), &artifact)?;
    if !complete {
        return Err(format!(
            "NCU incomplete; see {}",
            out.join("ncu.json").display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_live_details_export() {
        let bytes = include_bytes!("../tests/fixtures/ncu-details.csv");
        let launch = identity(&bytes[..], "ds4_moe_worklist_mmq_kernel").unwrap();
        assert_eq!(launch.grid, "(48, 1, 1)");
        assert_eq!(launch.block, "(32, 8, 1)");
        let values = metrics(&bytes[..]).unwrap();
        let duration = values
            .iter()
            .find(|m| m.name == "gpu__time_duration.sum")
            .unwrap();
        assert_eq!(duration.unit, "ns");
        assert_eq!(duration.value, Some(3101888.0));
        assert!(values
            .iter()
            .find(|m| m.name.starts_with("dram__"))
            .unwrap()
            .value
            .is_none());
    }

    #[test]
    fn requires_launch_identity() {
        assert!(metrics(
            b"Metric Name,Metric Unit,Metric Value\ngpu__time_duration.sum,nsecond,100\n"
                .as_slice()
        )
        .is_err());
    }
    #[test]
    fn counter_units_and_missing() {
        let m = metrics(b"ID,Metric Name,Metric Unit,Metric Value\n0,gpu__time_duration.sum,nsecond,\"1,234.5\"\n0,launch__block_size,thread,NA\n".as_slice()).unwrap();
        assert_eq!(
            m.iter()
                .find(|m| m.name == "gpu__time_duration.sum")
                .unwrap()
                .value,
            Some(1234.5)
        );
        assert_eq!(
            m.iter()
                .find(|m| m.name == "launch__block_size")
                .unwrap()
                .value,
            None
        );
        assert!(
            metrics(b"ID,Metric Name,Metric Unit,Metric Value\n0,x,,1\n1,x,,2\n".as_slice())
                .is_err()
        );
    }
}
