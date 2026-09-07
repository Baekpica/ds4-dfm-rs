use crate::{
    bench,
    doctor::Doctor,
    nsys::{Evidence, Kernel, Phase},
    runner,
};
use std::ffi::OsString;

// Scout v1 heuristics: execution-path triage, never a hardware bottleneck proof.
const HOST_MAX_COVERAGE: f64 = 0.75;
const HOST_MIN_GAP_NS: f64 = 100_000.0;
const MEM_MIN_BUSY_SHARE: f64 = 0.20;
const SHORT_KERNEL_NS: f64 = 10_000.0;
const SHORT_MIN_LAUNCHES: u64 = 100;
const SHORT_MIN_TIME_SHARE: f64 = 0.35;
const DOMINANT_MIN_SHARE: f64 = 0.15;
const TOP_KERNEL_LIMIT: usize = 10;

#[derive(Debug, PartialEq, Eq)]
pub enum Diagnosis {
    HostIdle,
    Memop,
    Fragmentation,
    Dominant,
    Mixed,
    Unknown,
}

impl Diagnosis {
    fn label(&self) -> &str {
        match self {
            Self::HostIdle => "HOST / IDLE",
            Self::Memop => "GPU MEMOP",
            Self::Fragmentation => "LAUNCH FRAGMENTATION",
            Self::Dominant => "DOMINANT KERNEL",
            Self::Mixed => "MIXED",
            Self::Unknown => "UNKNOWN",
        }
    }
    fn next(&self) -> &str {
        match self {
            Self::HostIdle => "inspect host-side PLE, routing, indexing, synchronization and GPU-free spans before NCU",
            Self::Memop => "inspect phase-local memcpy/memset, repeated copies and materialization before NCU",
            Self::Fragmentation => "inspect fusion, graph capture and repeated transforms before tuning tiles",
            Self::Dominant => "inspect this kernel's dispatch/fallback path; then use targeted NCU if the path is expected",
            Self::Mixed => "inspect repeated transforms, routing and kernel families across this phase",
            Self::Unknown => "inspect raw reports and missing phase/trace evidence before choosing a target",
        }
    }
}

fn ratio(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    a.zip(b)
        .filter(|(a, b)| a.is_finite() && b.is_finite() && *a >= 0.0 && *b > 0.0)
        .map(|(a, b)| a / b)
}

fn kernel_total(phase: &Phase) -> f64 {
    phase.kernels.iter().map(|k| k.total_ns).sum()
}

fn short_work(phase: &Phase) -> (u64, f64) {
    phase.kernels.iter().fold((0u64, 0.0), |(count, time), k| {
        if k.count
            .is_some_and(|n| n > 0 && k.total_ns / n as f64 <= SHORT_KERNEL_NS)
        {
            (
                count.saturating_add(k.count.unwrap_or(0)),
                time + k.total_ns,
            )
        } else {
            (count, time)
        }
    })
}

pub fn classify(phase: &Phase) -> Diagnosis {
    let Some(coverage) = ratio(phase.busy_ns, phase.wall_ns) else {
        return Diagnosis::Unknown;
    };
    if coverage > 1.0 || phase.mem_ns.is_none() || phase.gap_ns.is_none() {
        return Diagnosis::Unknown;
    }
    if coverage < HOST_MAX_COVERAGE && phase.gap_ns.unwrap_or(0.0) >= HOST_MIN_GAP_NS {
        return Diagnosis::HostIdle;
    }
    if ratio(phase.mem_ns, phase.busy_ns).is_some_and(|share| share >= MEM_MIN_BUSY_SHARE) {
        return Diagnosis::Memop;
    }
    let total = kernel_total(phase);
    if total <= 0.0 || !total.is_finite() {
        return Diagnosis::Unknown;
    }
    let (short_count, short_time) = short_work(phase);
    if short_count >= SHORT_MIN_LAUNCHES && short_time / total >= SHORT_MIN_TIME_SHARE {
        return Diagnosis::Fragmentation;
    }
    if phase
        .kernels
        .iter()
        .any(|k| k.total_ns / total >= DOMINANT_MIN_SHARE)
    {
        return Diagnosis::Dominant;
    }
    Diagnosis::Mixed
}

fn seconds(value: Option<f64>) -> String {
    value
        .map(|v| format!("{:.6} s", v / 1e9))
        .unwrap_or_else(|| "UNKNOWN".into())
}
fn percent(value: Option<f64>) -> String {
    value
        .map(|v| format!("{:.1} %", v * 100.0))
        .unwrap_or_else(|| "UNKNOWN".into())
}

pub fn render(
    d: &Doctor,
    command: &[OsString],
    rows: &[bench::Row],
    evidence: &Evidence,
) -> String {
    let mut out = format!("DS4 PERFORMANCE SCOUT\n\nWORKLOAD\n  command  {}\n  git      {} (dirty={})\n  gpu      {}\n\nAPPLICATION THROUGHPUT (fresh, unprofiled process)\n",
        runner::shell(command), d.facts["git_sha"], d.facts["git_dirty"], d.facts["gpu"]);
    if rows.is_empty() {
        out.push_str("  UNKNOWN: no recognized ds4-bench CSV on stdout\n");
    }
    for row in rows {
        out.push_str(&format!("  ctx {}: prefill {} tokens, {:.2} tok/s; decode {} tokens, {:.2} tok/s\n    first_token_sec {:.4}; kvcache_bytes {}\n", row.ctx, row.prefill_tokens, row.prefill_tps, row.gen_tokens, row.gen_tps, row.first_token_sec, row.kvcache_bytes));
    }
    out.push_str("\nEXECUTION-PATH STRUCTURE (separate profiled process)\n  Profiled throughput is informational and is never compared with baseline TPS.\n  Projection is a first-to-last GPU span, including gaps; it is not GPU busy time.\n  Coverage unions traced GPU intervals clipped to CPU NVTX phases.\n  Kernel percentages use summed kernel time within each phase; overlaps can double-count.\n");
    if rows.len() > 1 {
        out.push_str("  Multiple benchmark rows: phase reports aggregate frontiers; use one frontier for attribution.\n");
    }
    for name in crate::nsys::PHASES {
        let Some(p) = evidence.phases.get(name) else {
            out.push_str(&format!(
                "\n{name}: UNKNOWN (no phase evidence; may be absent from workload)\n"
            ));
            continue;
        };
        let coverage = ratio(p.busy_ns, p.wall_ns);
        let mem_share = ratio(p.mem_ns, p.busy_ns);
        out.push_str(&format!("\nNVTX GPU PROJECTION: {name}\n  range time      {}\n  projected span  {}\n  GPU busy union  {}\n  GPU coverage    {}\n  largest GPU-free span inside phase  {}\n  GPU memop union {} ({} of busy)\n\n{} TOP KERNELS\n", seconds(p.wall_ns), seconds(p.projected_ns), seconds(p.busy_ns), percent(coverage), seconds(p.gap_ns), seconds(p.mem_ns), percent(mem_share), name.to_uppercase()));
        top_kernels(&mut out, &p.kernels);
        let diagnosis = classify(p);
        let (short_count, short_time) = short_work(p);
        out.push_str(&format!("\nDIAGNOSIS  {}\nEVIDENCE\n  GPU coverage {}; largest gap {}\n  memop share {}; short-kernel launches {}, {} of kernel time (mean <= {:.0} us)\nNEXT\n  {}\n", diagnosis.label(), percent(coverage), seconds(p.gap_ns), percent(mem_share), short_count, percent(ratio(Some(short_time), Some(kernel_total(p)))), SHORT_KERNEL_NS / 1000.0, diagnosis.next()));
        if diagnosis == Diagnosis::Dominant {
            if let Some(kernel) = p.kernels.first() {
                out.push_str(&format!("\nNEXT NCU TARGET\n  kernel: {}\n  reason: {:.1} % of {name} summed kernel time\n", kernel.name, 100.0 * kernel.total_ns / kernel_total(p)));
                match ncu_command(&d.caps.ncu_help, name, &kernel.name, command) {
                    Some(cmd) => out.push_str(&format!("  suggested: {}\n  Profiles one matching launch in this NVTX phase. Review memory headroom and graph replay before running.\n  With a resident weight owner, prefer the matching model-free probe (see the optimization playbook).\n", runner::shell(&cmd))),
                    None => out.push_str("  suggested: unavailable; installed NCU lacks verified bounded kernel/NVTX filtering\n"),
                }
            }
        }
    }
    if !evidence.global.is_empty() {
        out.push_str(
            "\nGLOBAL CUDA KERNELS (all phases and setup; excluded from phase diagnosis)\n",
        );
        top_kernels(&mut out, &evidence.global);
    }
    out.push_str(&format!("\nHEURISTICS v1\n  host: coverage < {}%, gap >= {} us; memop: >= {}% busy;\n  fragmentation: >= {} short launches, >= {}% kernel time, mean <= {} us;\n  dominant: >= {}% summed phase kernel time. Missing structural evidence => UNKNOWN.\n", HOST_MAX_COVERAGE*100.0, HOST_MIN_GAP_NS/1000.0, MEM_MIN_BUSY_SHARE*100.0, SHORT_MIN_LAUNCHES, SHORT_MIN_TIME_SHARE*100.0, SHORT_KERNEL_NS/1000.0, DOMINANT_MIN_SHARE*100.0));
    out.push_str("\nLIMITS\n  Nsight Systems does not establish memory-bound versus compute-bound behavior.\n  A profiler result is not a correctness proof or a claimed performance improvement.\n  Return every code change to same-workload, same-hour A/B and existing correctness gates.\n  Cache warmth, clocks, binary/model/prompt identity and unrecorded environment still require review.\n");
    for warning in d.warnings.iter().chain(&evidence.warnings) {
        out.push_str(&format!("  WARNING: {warning}\n"));
    }
    out
}

fn top_kernels(out: &mut String, kernels: &[Kernel]) {
    let total: f64 = kernels.iter().map(|k| k.total_ns).sum();
    for k in kernels.iter().take(TOP_KERNEL_LIMIT) {
        out.push_str(&format!(
            "  {}  {}  launches={}  {}\n",
            percent(ratio(Some(k.total_ns), Some(total))),
            seconds(Some(k.total_ns)),
            k.count
                .map(|n| n.to_string())
                .unwrap_or_else(|| "UNKNOWN".into()),
            k.name
        ));
    }
}

fn ncu_command(help: &str, phase: &str, name: &str, bench: &[OsString]) -> Option<Vec<OsString>> {
    if ![
        "--kernel-name-base",
        "--kernel-name",
        "--launch-count",
        "--nvtx-include",
        "--nvtx",
    ]
    .iter()
    .all(|flag| help.contains(flag))
    {
        return None;
    }
    let base = if name.contains(['<', '(']) {
        "demangled"
    } else {
        "function"
    };
    let mut regex = String::from("regex:^");
    for c in name.chars() {
        if ".+*?()[]{}^$|\\".contains(c) {
            regex.push('\\');
        }
        regex.push(c);
    }
    regex.push('$');
    let mut command: Vec<OsString> = [
        "ncu",
        "--kernel-name-base",
        base,
        "--kernel-name",
        &regex,
        "--launch-count",
        "1",
        "--nvtx",
        "--nvtx-include",
        &format!("{phase}/"),
    ]
    .map(Into::into)
    .into();
    command.extend_from_slice(bench);
    Some(command)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase(shares: &[f64]) -> Phase {
        Phase {
            name: "ds4.prefill".into(),
            wall_ns: Some(1e9),
            busy_ns: Some(0.95e9),
            mem_ns: Some(0.0),
            gap_ns: Some(0.01e9),
            kernels: shares
                .iter()
                .enumerate()
                .map(|(i, s)| Kernel {
                    name: format!("kernel{i}"),
                    total_ns: s * 0.95e9,
                    count: Some(1),
                })
                .collect(),
            ..Default::default()
        }
    }
    #[test]
    fn classifier_priority() {
        let mut p = phase(&[0.8, 0.2]);
        assert_eq!(classify(&p), Diagnosis::Dominant);
        p.busy_ns = Some(0.6e9);
        assert_eq!(classify(&p), Diagnosis::HostIdle);
        p.busy_ns = Some(0.95e9);
        p.mem_ns = Some(0.3e9);
        assert_eq!(classify(&p), Diagnosis::Memop);
        p.mem_ns = Some(0.0);
        p.kernels[0].count = Some(100_000);
        assert_eq!(classify(&p), Diagnosis::Fragmentation);
        assert_eq!(classify(&phase(&[0.1; 10])), Diagnosis::Mixed);
        p.busy_ns = None;
        assert_eq!(classify(&p), Diagnosis::Unknown);
        assert_eq!(classify(&Phase::default()), Diagnosis::Unknown);
    }
    #[test]
    fn ncu_is_phase_and_launch_limited() {
        let help = "--kernel-name --kernel-name-base --launch-count --nvtx --nvtx-include";
        let cmd = ncu_command(
            help,
            "ds4.decode",
            "foo<int, 128>",
            &["./ds4-bench".into(), "a'$(b)".into()],
        )
        .unwrap();
        assert!(cmd.windows(2).any(|w| w == ["--launch-count", "1"]));
        assert!(cmd
            .windows(2)
            .any(|w| w == ["--nvtx-include", "ds4.decode/"]));
        assert!(!cmd.iter().any(|s| s == "full"));
        assert!(ncu_command("", "ds4.decode", "foo", &[]).is_none());
    }
}
