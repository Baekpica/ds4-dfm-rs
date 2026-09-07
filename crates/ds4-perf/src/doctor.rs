use crate::runner::{self, Capture, Probe};
use std::collections::BTreeMap;

#[derive(Default, Debug)]
pub struct Capabilities {
    pub profile_help: String,
    pub reports: String,
    pub sync_rule: bool,
    pub ncu_help: String,
}

impl Capabilities {
    pub fn report(&self, name: &str) -> Option<String> {
        let base = name.split(':').next()?;
        let line = self.reports.lines().find(|line| {
            line.trim_start()
                .strip_prefix(base)
                .is_some_and(|rest| rest.starts_with([' ', '[']))
        })?;
        let mut selected = base.to_string();
        for option in name.split(':').skip(1) {
            if line.contains(&format!(":{option}")) {
                selected.push(':');
                selected.push_str(option);
            }
        }
        Some(selected)
    }

    pub fn can_scout(&self) -> bool {
        self.profile_help.contains("cuda")
            && self.profile_help.contains("nvtx")
            && (self.report("nvtx_kern_sum").is_some()
                || self.report("cuda_gpu_kern_sum").is_some())
    }

    pub fn graph_nodes(&self) -> bool {
        self.profile_help.contains("--cuda-graph-trace") && self.profile_help.contains("'node'")
    }
}

pub struct Doctor {
    pub facts: BTreeMap<String, String>,
    pub caps: Capabilities,
    pub warnings: Vec<String>,
    pub probes: Vec<(String, Capture)>,
}

pub fn inspect(probe: &mut impl Probe, benchmark: Option<&str>) -> Doctor {
    let mut probes = Vec::new();
    let mut run = |program: &str, args: &[&str]| {
        let result = probe.capture(program, args);
        probes.push((format!("{program} {}", args.join(" ")), result.clone()));
        result
    };
    let mut facts = BTreeMap::new();
    facts.insert("os".into(), std::env::consts::OS.into());
    facts.insert("arch".into(), std::env::consts::ARCH.into());
    facts.insert(
        "git_sha".into(),
        summary(&run("git", &["rev-parse", "HEAD"])),
    );
    let dirty = run(
        "git",
        &["status", "--porcelain", "--untracked-files=normal"],
    );
    facts.insert(
        "git_dirty".into(),
        if dirty.ok {
            (!dirty.out.trim().is_empty()).to_string()
        } else {
            "unknown".into()
        },
    );
    facts.insert("hostname".into(), summary(&run("hostname", &[])));
    let smi = run("nvidia-smi", &[]);
    facts.insert(
        "nvidia-smi".into(),
        if smi.ok { "OK" } else { "unavailable" }.into(),
    );
    let gpu = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader",
        ],
    );
    let mut gpu_rows = Vec::new();
    if gpu.ok {
        let _ = crate::csv::records(gpu.out.as_bytes(), |row| {
            gpu_rows.push(row);
            Ok(())
        });
    }
    // Older drivers may not expose compute_cap; retain the name/driver anyway.
    if gpu_rows.is_empty() {
        let gpu = run(
            "nvidia-smi",
            &["--query-gpu=name,driver_version", "--format=csv,noheader"],
        );
        if gpu.ok {
            let _ = crate::csv::records(gpu.out.as_bytes(), |mut row| {
                if row.len() == 2 {
                    row.insert(1, "unknown".into());
                }
                gpu_rows.push(row);
                Ok(())
            });
        }
    }
    for (name, index) in [("gpu", 0), ("compute", 1), ("driver", 2)] {
        let values: Vec<_> = gpu_rows
            .iter()
            .filter_map(|row| row.get(index))
            .map(|s| s.trim().to_string())
            .collect();
        facts.insert(
            name.into(),
            if values.is_empty() {
                "unknown".into()
            } else {
                values.join("; ")
            },
        );
    }
    facts.insert("cuda".into(), summary(&run("nvcc", &["--version"])));
    facts.insert(
        "cuda_driver_api".into(),
        smi.out
            .lines()
            .find(|line| line.contains("CUDA") && line.contains("Version"))
            .unwrap_or("unknown")
            .trim()
            .into(),
    );
    facts.insert("nsys".into(), summary(&run("nsys", &["--version"])));
    let profile = run("nsys", &["profile", "--help"]);
    let reports = run("nsys", &["stats", "--help-reports"]);
    let rules = run("nsys", &["analyze", "--help-rules"]);
    facts.insert("ncu".into(), summary(&run("ncu", &["--version"])));
    let ncu = run("ncu", &["--help"]);
    let bench = benchmark.map(|program| run(program, &["--help"]));
    facts.insert(
        "benchmark".into(),
        benchmark.unwrap_or("supplied scout command").into(),
    );
    facts.insert("NVTX".into(), "unknown (not probed)".into());

    let caps = Capabilities {
        profile_help: if profile.ok {
            profile.combined()
        } else {
            String::new()
        },
        // Nsight 2026.1 exits 1 after successfully listing reports/rules.
        // Accept the documented help body, not arbitrary failed-command text.
        reports: if reports.ok
            || reports
                .combined()
                .contains("The following built-in reports are available:")
        {
            reports.combined()
        } else {
            String::new()
        },
        sync_rule: (rules.ok
            || rules
                .combined()
                .contains("The following built-in reports are available:"))
            && rules
                .combined()
                .lines()
                .any(|l| l.trim_start().starts_with("cuda_api_sync[")),
        ncu_help: if ncu.ok {
            ncu.combined()
        } else {
            String::new()
        },
    };
    let mut warnings = Vec::new();
    if !smi.ok {
        warnings.push("nvidia-smi unavailable; GPU/driver readiness unknown".into());
    }
    if !caps.can_scout() {
        warnings
            .push("nsys lacks CUDA/NVTX collection or kernel reports; scout unavailable".into());
    }
    if !caps.graph_nodes() {
        warnings.push(
            "CUDA graph-node tracing unavailable; captured decode may lack kernel detail".into(),
        );
    }
    for report in [
        "nvtx_gpu_proj_sum",
        "nvtx_kern_sum",
        "nvtx_pushpop_trace",
        "cuda_gpu_trace",
        "cuda_gpu_mem_time_sum",
    ] {
        if caps.report(report).is_none() {
            warnings.push(format!(
                "{report} unavailable; related scout evidence will be UNKNOWN"
            ));
        }
    }
    if !caps.sync_rule && caps.report("cuda_api_sync").is_none() {
        warnings.push("cuda_api_sync unavailable; scout omits synchronization analysis".into());
    }
    if caps.ncu_help.is_empty() {
        warnings.push("ncu unavailable; scout works, kernel deep-dive unavailable".into());
    }
    let mut doctor = Doctor {
        facts,
        caps,
        warnings,
        probes,
    };
    if let Some(bench) = bench {
        if bench.ok {
            doctor.benchmark(&bench.combined(), "NVTX: ");
        } else {
            doctor
                .warnings
                .push("target benchmark --help failed; NVTX capability unknown".into());
        }
    }
    doctor
}

fn summary(result: &Capture) -> String {
    if !result.ok {
        return "unavailable".into();
    }
    let text = result.combined();
    let lines: Vec<_> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    lines.join(" | ")
}

impl Doctor {
    // Only the named target's help or the actual baseline's stderr supplies this fact.
    pub fn benchmark(&mut self, output: &str, prefix: &str) {
        let value = output.lines().find_map(|line| line.strip_prefix(prefix));
        let capability = match value {
            Some("available") => "available (ds4.prefill / ds4.decode)",
            Some(s) if s.starts_with("unavailable") => {
                "unavailable (build with make ds4-bench-perf)"
            }
            _ => "unknown (benchmark has no capability marker)",
        };
        self.facts.insert("NVTX".into(), capability.into());
        if value != Some("available") {
            self.warnings.push(format!(
                "benchmark NVTX {capability}; phase diagnosis requires ds4 ranges"
            ));
        }
    }

    pub fn render(&self) -> String {
        let mut out = String::from("ds4-perf doctor\n\n");
        for name in [
            "git_sha",
            "git_dirty",
            "hostname",
            "os",
            "arch",
            "gpu",
            "compute",
            "driver",
            "cuda",
            "cuda_driver_api",
            "nsys",
            "ncu",
            "benchmark",
            "NVTX",
        ] {
            out.push_str(&format!("{name:<16} {}\n", self.facts[name]));
        }
        out.push_str(&format!(
            "status           {}\n",
            if self.warnings.is_empty() {
                "READY"
            } else {
                "DEGRADED"
            }
        ));
        for warning in &self.warnings {
            out.push_str(&format!("reason/impact    {warning}\n"));
        }
        out
    }

    pub fn save_probes(&self, out: &std::path::Path) -> Result<(), String> {
        for (i, (command, result)) in self.probes.iter().enumerate() {
            runner::write(
                &out.join(format!("probe-{i:02}.txt")),
                &format!(
                    "command: {command}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
                    result.ok, result.out, result.err
                ),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake;
    impl Probe for Fake {
        fn capture(&mut self, program: &str, args: &[&str]) -> Capture {
            let out = match (program, args) {
                ("nsys", ["profile", "--help"]) => "--trace cuda nvtx --cuda-graph-trace 'node' --sample --cpuctxsw",
                ("nsys", ["stats", "--help-reports"]) => "  cuda_gpu_kern_sum[:nvtx-name][:base|:mangled] -- kernels\n  nvtx_kern_sum[:base|:mangled] -- phases",
                ("nvidia-smi", [_, _]) => "NVIDIA GB10, 12.1, 610.43.02",
                ("./ds4-bench", _) => "help\nNVTX: available\n",
                ("ncu", _) => return Capture::failed("not found"),
                _ => "version",
            };
            Capture {
                ok: true,
                out: out.into(),
                err: String::new(),
            }
        }
    }

    #[test]
    fn missing_ncu_keeps_scout() {
        let d = inspect(&mut Fake, Some("./ds4-bench"));
        assert!(d.caps.can_scout());
        assert!(d.caps.graph_nodes());
        assert_eq!(d.facts["gpu"], "NVIDIA GB10");
        assert_eq!(
            d.caps.report("cuda_gpu_kern_sum:nvtx-name:base").as_deref(),
            Some("cuda_gpu_kern_sum:nvtx-name:base")
        );
        assert!(d.caps.report("nvtx_gpu_proj_sum").is_none());
        assert!(d.render().contains("DEGRADED"));
        assert!(d
            .render()
            .contains("scout works, kernel deep-dive unavailable"));
    }

    #[test]
    fn older_report_drops_options() {
        let caps = Capabilities {
            reports: "cuda_gpu_kern_sum -- summary\n".into(),
            ..Default::default()
        };
        assert_eq!(
            caps.report("cuda_gpu_kern_sum:nvtx-name:base").as_deref(),
            Some("cuda_gpu_kern_sum")
        );
        assert!(!caps.can_scout());
    }

    #[test]
    fn help_exit_one_is_supported() {
        struct HelpExitOne;
        impl Probe for HelpExitOne {
            fn capture(&mut self, program: &str, args: &[&str]) -> Capture {
                let mut result = Fake.capture(program, args);
                if program == "nsys" && args == ["stats", "--help-reports"] {
                    result.ok = false;
                    result.out = format!(
                        "The following built-in reports are available:\n{}",
                        result.out
                    );
                }
                result
            }
        }
        assert!(inspect(&mut HelpExitOne, None).caps.can_scout());
    }
}
