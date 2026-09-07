use crate::{
    csv,
    nsys::{phase_name, Phase},
};
use std::collections::{BTreeMap, BTreeSet};
use std::io::BufRead;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
struct Span {
    start: f64,
    end: f64,
}

#[derive(Default)]
struct Window {
    ranges: Vec<Span>,
    busy: Vec<Span>,
    memory: Vec<Span>,
}

pub fn load(out: &Path, phases: &mut BTreeMap<String, Phase>) -> Result<(), String> {
    let ranges = csv::open(&out.join("nsys-ranges.csv")).map_err(|e| e.to_string())?;
    let gpu = csv::open(&out.join("nsys-gpu-trace.csv")).map_err(|e| e.to_string())?;
    analyze(ranges, gpu, phases)
}

fn analyze(
    ranges: impl BufRead,
    gpu: impl BufRead,
    phases: &mut BTreeMap<String, Phase>,
) -> Result<(), String> {
    let mut windows: BTreeMap<String, Window> = BTreeMap::new();
    let mut threads = BTreeSet::new();
    csv::table(ranges, "name", |header, row| {
        let Some(name) = csv::value(row, header, &["name"]).and_then(phase_name) else {
            return Ok(());
        };
        let start = csv::time(row, header, "Start").ok_or("range Start missing")?;
        let duration = csv::time(row, header, "Duration").ok_or("range Duration missing")?;
        let pid = csv::value(row, header, &["pid"]).ok_or("range PID missing")?;
        let tid = csv::value(row, header, &["tid"]).ok_or("range TID missing")?;
        threads.insert((pid.to_string(), tid.to_string()));
        windows.entry(name.into()).or_default().ranges.push(Span {
            start,
            end: start + duration,
        });
        Ok(())
    })?;
    if windows.is_empty() || threads.len() != 1 {
        return Err("need complete phase ranges from one benchmark thread".into());
    }
    let mut all_ranges: Vec<_> = windows
        .values()
        .flat_map(|w| w.ranges.iter())
        .copied()
        .collect();
    all_ranges.sort_by(|a, b| a.start.total_cmp(&b.start));
    if all_ranges.windows(2).any(|p| p[0].end > p[1].start) {
        return Err("overlapping benchmark phases".into());
    }
    let mut devices = BTreeSet::new();
    let mut events = 0;
    csv::table(gpu, "name", |header, row| {
        let start = csv::time(row, header, "Start").ok_or("GPU Start missing")?;
        let duration = csv::time(row, header, "Duration").ok_or("GPU Duration missing")?;
        let device = csv::value(row, header, &["device"]).ok_or("GPU Device missing")?;
        let ctx = csv::value(row, header, &["ctx"]).ok_or("GPU Ctx missing")?;
        let name = csv::value(row, header, &["name"]).ok_or("GPU Name missing")?;
        // Nsight's formatter chooses B/KB/MB dynamically. Only presence,
        // not byte magnitude, identifies memory operations in this timeline.
        let bytes = header
            .iter()
            .position(|s| csv::key(s).starts_with("bytes ("))
            .ok_or("GPU memory-operation column missing")?;
        devices.insert((device.to_string(), ctx.to_string()));
        events += 1;
        let memory = row.get(bytes).is_some_and(|s| !s.trim().is_empty())
            || name.starts_with("[CUDA memcpy")
            || name.starts_with("[CUDA memset");
        for window in windows.values_mut() {
            for range in &window.ranges {
                let span = Span {
                    start: start.max(range.start),
                    end: (start + duration).min(range.end),
                };
                if span.end <= span.start {
                    continue;
                }
                window.busy.push(span);
                if memory {
                    window.memory.push(span);
                }
            }
        }
        Ok(())
    })?;
    // Missing traces must never become "0% GPU coverage". Multiple devices
    // or contexts also lack a defensible single-GPU idle denominator.
    if events == 0 || devices.len() != 1 {
        return Err("empty or multiple-device/context GPU trace".into());
    }
    for (name, window) in windows {
        let ranges = union(window.ranges);
        let busy = union(window.busy);
        let memory = union(window.memory);
        let mut largest: f64 = 0.0;
        for range in &ranges {
            let mut cursor = range.start;
            for span in &busy {
                if span.end <= range.start || span.start >= range.end {
                    continue;
                }
                largest = largest.max(span.start - cursor);
                cursor = cursor.max(span.end);
            }
            largest = largest.max(range.end - cursor);
        }
        let phase = phases.entry(name.clone()).or_insert_with(|| Phase {
            name,
            ..Default::default()
        });
        phase.wall_ns = Some(duration(&ranges));
        phase.busy_ns = Some(duration(&busy));
        phase.mem_ns = Some(duration(&memory));
        phase.gap_ns = Some(largest);
    }
    Ok(())
}

fn duration(spans: &[Span]) -> f64 {
    spans.iter().map(|s| s.end - s.start).sum()
}

fn union(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut merged: Vec<Span> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut() {
            if span.start <= last.end {
                last.end = last.end.max(span.end);
                continue;
            }
        }
        merged.push(span);
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nsight_scaled_memory_units() {
        for unit in ["MB", "KB", "GiB"] {
            let gpu = include_str!("../tests/fixtures/gpu-trace.csv")
                .replace("Bytes (B)", &format!("Bytes ({unit})"));
            let mut phases = BTreeMap::new();
            analyze(
                include_bytes!("../tests/fixtures/ranges.csv").as_slice(),
                gpu.as_bytes(),
                &mut phases,
            )
            .unwrap();
            assert_eq!(phases["ds4.prefill"].busy_ns, Some(600.0));
            assert_eq!(phases["ds4.prefill"].mem_ns, Some(100.0));
        }
    }

    #[test]
    fn clips_unions_and_keeps_gaps() {
        let mut phases = BTreeMap::new();
        analyze(
            include_bytes!("../tests/fixtures/ranges.csv").as_slice(),
            include_bytes!("../tests/fixtures/gpu-trace.csv").as_slice(),
            &mut phases,
        )
        .unwrap();
        let p = &phases["ds4.prefill"];
        assert_eq!(p.wall_ns, Some(1000.0));
        assert_eq!(p.busy_ns, Some(600.0));
        assert_eq!(p.mem_ns, Some(100.0));
        assert_eq!(p.gap_ns, Some(300.0));
        assert_eq!(phases["ds4.decode"].busy_ns, Some(800.0));
        assert!(analyze(
            include_bytes!("../tests/fixtures/ranges.csv").as_slice(),
            b"".as_slice(),
            &mut BTreeMap::new()
        )
        .is_err());
    }
}
