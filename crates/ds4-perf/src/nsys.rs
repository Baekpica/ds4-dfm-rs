use crate::{csv, doctor::Capabilities};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::BufRead;
use std::path::Path;

pub const PHASES: [&str; 2] = ["ds4.prefill", "ds4.decode"];

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    pub name: String,
    pub total_ns: f64,
    pub count: Option<u64>,
}

#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phase {
    pub name: String,
    pub wall_ns: Option<f64>,
    pub projected_ns: Option<f64>,
    pub busy_ns: Option<f64>,
    pub mem_ns: Option<f64>,
    pub gap_ns: Option<f64>,
    pub kernels: Vec<Kernel>,
}

#[derive(Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub phases: BTreeMap<String, Phase>,
    pub global: Vec<Kernel>,
    pub warnings: Vec<String>,
}

pub fn phase_name(name: &str) -> Option<&'static str> {
    let name = name.strip_prefix(':').unwrap_or(name);
    PHASES.into_iter().find(|p| *p == name)
}

pub fn profile_command(caps: &Capabilities, out: &Path, bench: &[OsString]) -> Vec<OsString> {
    let mut args: Vec<OsString> = ["nsys", "profile", "--trace=cuda,nvtx"]
        .map(Into::into)
        .into();
    for option in ["--sample", "--cpuctxsw"] {
        if caps.profile_help.contains(option) {
            args.push(format!("{option}=none").into());
        }
    }
    if caps.graph_nodes() {
        args.push("--cuda-graph-trace=node".into());
    }
    args.push("--output".into());
    args.push(out.join("trace").into());
    args.extend_from_slice(bench);
    args
}

pub fn collect_prepared(
    caps: &Capabilities,
    out: &Path,
    prepared: &crate::experiment::Prepared,
) -> Evidence {
    collect_with(caps, out, |cmd, out, name| prepared.run(cmd, out, name))
}

fn collect_with(
    caps: &Capabilities,
    out: &Path,
    mut run: impl FnMut(&[OsString], &Path, &str) -> Result<(), String>,
) -> Evidence {
    let mut evidence = Evidence::default();
    let mut input = out.join("trace.nsys-rep");
    // Keep each stdout/stderr verbatim. In particular cuda_api_sync is an
    // analyze rule in newer Nsight, not a stats report.
    for (requested, file) in [
        ("nvtx_gpu_proj_sum", "nsys-nvtx"),
        ("nvtx_kern_sum:base", "nsys-nvtx-kernels"),
        // Keep global CUDA kernels even when NVTX_EVENTS is absent.
        ("cuda_gpu_kern_sum:base", "nsys-kernels"),
        ("cuda_gpu_mem_time_sum", "nsys-memory"),
        ("nvtx_pushpop_trace", "nsys-ranges"),
        ("cuda_gpu_trace:base", "nsys-gpu-trace"),
        ("cuda_api_sync", "nsys-sync"),
    ] {
        let rule = requested == "cuda_api_sync" && caps.sync_rule;
        let selected = if rule {
            Some(requested.into())
        } else {
            caps.report(requested)
        };
        let Some(selected) = selected else {
            evidence
                .warnings
                .push(format!("{requested}: unsupported by installed nsys"));
            continue;
        };
        let mut command: Vec<OsString> = [
            "nsys",
            if rule { "analyze" } else { "stats" },
            "--format",
            "csv:noconv",
            if rule { "--rule" } else { "--report" },
        ]
        .map(Into::into)
        .into();
        command.push(selected.into());
        command.push(input.clone().into());
        match run(&command, out, file) {
            Ok(()) => {
                // Reuse the completed export. Reopening the .nsys-rep can
                // trigger Nsight's stale-export check between reports.
                let sqlite = out.join("trace.sqlite");
                if sqlite.is_file() {
                    input = sqlite;
                }
                if let Err(err) = std::fs::copy(
                    out.join(format!("{file}.stdout")),
                    out.join(format!("{file}.csv")),
                ) {
                    evidence.warnings.push(err.to_string());
                }
            }
            Err(err) => evidence.warnings.push(err),
        }
    }
    load(out, evidence)
}

fn load(out: &Path, mut evidence: Evidence) -> Evidence {
    for file in ["nsys-nvtx-kernels", "nsys-kernels"] {
        let Ok(reader) = csv::open(&out.join(format!("{file}.csv"))) else {
            continue;
        };
        match kernels(reader) {
            Ok(groups) => {
                for (name, kernels) in groups {
                    if name.is_empty() {
                        evidence.global = kernels;
                        continue;
                    }
                    let phase = evidence
                        .phases
                        .entry(name.clone())
                        .or_insert_with(|| Phase {
                            name,
                            ..Default::default()
                        });
                    // Prefer explicit NVTX association; never add the same report twice.
                    if phase.kernels.is_empty() {
                        phase.kernels = kernels;
                    }
                }
            }
            Err(err) => evidence.warnings.push(format!("{file}: {err}")),
        }
    }
    if let Ok(reader) = csv::open(&out.join("nsys-nvtx.csv")) {
        if let Err(err) = projection(reader, &mut evidence.phases) {
            evidence.warnings.push(format!("nsys-nvtx: {err}"));
        }
    }
    match crate::timeline::load(out, &mut evidence.phases) {
        Ok(()) => {}
        Err(err) => evidence
            .warnings
            .push(format!("GPU busy coverage unknown: {err}")),
    }
    if evidence.phases.is_empty() {
        evidence.warnings.push(
            "CUDA-only analysis: no ds4.prefill / ds4.decode NVTX evidence; phase diagnosis UNKNOWN; build with make ds4-bench-perf".into(),
        );
    }
    evidence
}

pub fn kernels(reader: impl BufRead) -> Result<BTreeMap<String, Vec<Kernel>>, String> {
    let mut grouped: BTreeMap<(String, String), Kernel> = BTreeMap::new();
    let mut header = Vec::new();
    csv::records(reader, |row| {
        if csv::column(&row, &["kernel name", "name"]).is_some()
            && row.iter().any(|s| csv::key(s).starts_with("total time ("))
        {
            header = row;
            return Ok(());
        }
        if header.is_empty() || row.iter().all(|s| s.trim().is_empty()) {
            return Ok(());
        }
        let name =
            csv::value(&row, &header, &["kernel name", "name"]).ok_or("missing kernel name")?;
        let (phase, name) = if let Some(range) = csv::value(&row, &header, &["nvtx range", "range"])
        {
            (phase_name(range).unwrap_or(""), name)
        } else if let Some((range, kernel)) = name.split_once('/') {
            match phase_name(range) {
                Some(phase) => (phase, kernel),
                None => ("", name),
            }
        } else {
            ("", name)
        };
        let time = csv::time(&row, &header, "Total Time").ok_or("missing/invalid Total Time")?;
        let count = csv::value(&row, &header, &["kern inst", "instances"])
            .map(|v| v.parse::<u64>())
            .transpose()
            .map_err(|_| "invalid kernel count")?;
        let entry = grouped
            .entry((phase.into(), name.into()))
            .or_insert_with(|| Kernel {
                name: name.into(),
                count: Some(0),
                ..Default::default()
            });
        entry.total_ns += time;
        entry.count = entry.count.zip(count).and_then(|(a, b)| a.checked_add(b));
        Ok(())
    })?;
    let mut groups: BTreeMap<String, Vec<Kernel>> = BTreeMap::new();
    for ((phase, _), kernel) in grouped {
        groups.entry(phase).or_default().push(kernel);
    }
    for kernels in groups.values_mut() {
        kernels.sort_by(|a, b| b.total_ns.total_cmp(&a.total_ns).then(a.name.cmp(&b.name)));
    }
    Ok(groups)
}

fn projection(reader: impl BufRead, phases: &mut BTreeMap<String, Phase>) -> Result<(), String> {
    csv::table(reader, "range", |header, row| {
        let Some(name) = csv::value(row, header, &["range"]).and_then(phase_name) else {
            return Ok(());
        };
        let phase = phases.entry(name.into()).or_insert_with(|| Phase {
            name: name.into(),
            ..Default::default()
        });
        if let Some(time) = csv::time(row, header, "Total Range Time") {
            *phase.wall_ns.get_or_insert(0.0) += time;
        }
        if let Some(time) = csv::time(row, header, "Total Proj Time") {
            *phase.projected_ns.get_or_insert(0.0) += time;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_reuse_export() {
        let out = std::env::temp_dir().join(format!("ds4-nsys-export-{}", std::process::id()));
        std::fs::create_dir(&out).unwrap();
        let caps = Capabilities {
            reports: "nvtx_gpu_proj_sum -- projection\nnvtx_kern_sum[:base] -- kernels\ncuda_gpu_kern_sum[:base] -- kernels\n".into(),
            ..Default::default()
        };
        let mut completed = Vec::new();
        let evidence = collect_with(&caps, &out, |command, dir, name| {
            let input = Path::new(command.last().unwrap());
            let sqlite = dir.join("trace.sqlite");
            // Reproduce Nsight rejecting a second .nsys-rep read as stale,
            // even though the first report just completed its SQLite export.
            if sqlite.exists() && input != sqlite {
                return Err("Existing SQLite export found: older than input file".into());
            }
            std::fs::write(sqlite, []).unwrap();
            let csv = if name == "nsys-nvtx-kernels" {
                "Name,Total Time (ns)\n:ds4.decode/kernel,3000\n"
            } else {
                "Name,Total Time (ns)\nkernel,3000\n"
            };
            std::fs::write(dir.join(format!("{name}.stdout")), csv).unwrap();
            completed.push(name.to_owned());
            Ok(())
        });
        std::fs::remove_dir_all(out).unwrap();
        assert_eq!(
            completed,
            ["nsys-nvtx", "nsys-nvtx-kernels", "nsys-kernels"]
        );
        assert_eq!(evidence.phases["ds4.decode"].kernels[0].total_ns, 3000.0);
        assert_eq!(evidence.global[0].name, "kernel");
    }

    #[test]
    fn templates_repeats_and_phases() {
        let groups = kernels(include_bytes!("../tests/fixtures/kernels.csv").as_slice()).unwrap();
        let prefill = &groups["ds4.prefill"];
        assert_eq!(prefill[0].name, "foo<int, 128, bar<float>>");
        assert_eq!(prefill[0].total_ns, 3000.0);
        assert_eq!(prefill[0].count, Some(3));
        assert_eq!(groups["ds4.decode"][0].name, "ordinary_kernel");
    }

    #[test]
    fn older_and_empty_reports() {
        let groups = kernels(b"Name,Total Time (us)\na/b,3\n".as_slice()).unwrap();
        assert_eq!(groups[""][0].name, "a/b");
        let groups =
            kernels(b"Name,Total Time (us)\n:ds4.decode/ordinary_kernel,3\n".as_slice()).unwrap();
        assert_eq!(groups["ds4.decode"][0].count, None);
        assert_eq!(groups["ds4.decode"][0].total_ns, 3000.0);
        for text in [
            "",
            "SKIPPED: no CUDA kernel data\n",
            "UNKNOWN REPORT: nvtx_kern_sum\n",
        ] {
            assert!(kernels(text.as_bytes()).unwrap().is_empty());
        }
    }
}
