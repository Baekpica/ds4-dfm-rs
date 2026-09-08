use crate::{
    csv,
    nsys::{Evidence, Kernel, Phase},
    runner,
};
use ds4_perf::trace::Event;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

pub fn capture(
    out: &Path,
    args: &crate::cli::Scout,
    controls: &BTreeMap<std::ffi::OsString, std::ffi::OsString>,
) -> Result<Evidence, String> {
    let helper = crate::inspect::helper(args.gpu_helper.as_deref());
    let library = args
        .cupti_library
        .clone()
        .or_else(|| helper.map(|p| p.with_file_name("libds4_perf_gpu.so")))
        .ok_or("CUPTI collector unavailable; build make ds4-perf-gpu or pass --cupti-library")?
        .canonicalize()
        .map_err(|e| format!("CUPTI collector library: {e}"))?;
    let sdk = args
        .cupti_sdk
        .clone()
        .or_else(|| {
            runner::resolve("nvcc".as_ref()).and_then(|p| {
                p.parent()?
                    .parent()
                    .map(|p| p.join("extras/CUPTI/lib64/libcupti.so"))
            })
        })
        .ok_or("NVIDIA libcupti.so not found; pass --cupti-sdk")?
        .canonicalize()
        .map_err(|e| format!("NVIDIA CUPTI library: {e}"))?;
    let before = ds4_perf::artifact::hash(&library)?;
    let config = crate::experiment::Replay {
        helper: None,
        calibration: None,
        collector: Some(crate::experiment::InputFile {
            path: library.clone(),
            sha256: before.clone(),
        }),
        sdk: Some(crate::experiment::InputFile {
            path: sdk.clone(),
            sha256: ds4_perf::artifact::hash(&sdk)?,
        }),
    };
    std::fs::write(
        out.join("collector-config.json"),
        serde_json::to_vec_pretty(&config).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    runner::write(
        &out.join("collector.txt"),
        &format!(
            "collector={}\ncollector_sha256={}\nsdk={}\nsdk_sha256={}\n",
            library.display(),
            before,
            sdk.display(),
            ds4_perf::artifact::hash(&sdk)?
        ),
    )?;
    let mut env = controls.clone();
    env.insert(
        "CUDA_INJECTION64_PATH".into(),
        library.clone().into_os_string(),
    );
    env.insert("NVTX_INJECTION64_PATH".into(), sdk.clone().into_os_string());
    env.insert(
        "DS4_PERF_CUPTI_OUTPUT".into(),
        out.join("activity.jsonl").into_os_string(),
    );
    let mut paths = vec![sdk
        .parent()
        .ok_or("CUPTI library has no parent")?
        .to_path_buf()];
    if let Some(inherited) = controls.get(std::ffi::OsStr::new("LD_LIBRARY_PATH")) {
        paths.extend(std::env::split_paths(inherited));
    }
    env.insert(
        "LD_LIBRARY_PATH".into(),
        std::env::join_paths(paths).map_err(|e| e.to_string())?,
    );
    eprintln!("ds4-perf: fresh CUPTI-instrumented process");
    runner::run_limited(&args.command, out, "profile", &env, args.budget.limits())?;
    for input in [&config.collector, &config.sdk].into_iter().flatten() {
        if ds4_perf::artifact::hash(&input.path)? != input.sha256 {
            return Err("CUPTI collector or SDK changed during capture".into());
        }
    }
    collect(out)
}

// Normalize owned activity records only after checking the collector footer.
// Subtract the integer epoch before conversion: CUPTI timestamps exceed f64's
// exact integer range, whereas the duration of a bounded capture does not.
pub fn collect(out: &Path) -> Result<Evidence, String> {
    let input = BufReader::new(File::open(out.join("activity.jsonl")).map_err(|e| e.to_string())?);
    let (ranges, gpu, mut evidence) = normalize(input)?;
    runner::write(&out.join("nsys-ranges.csv"), &ranges)?;
    runner::write(&out.join("nsys-gpu-trace.csv"), &gpu)?;
    crate::timeline::load(out, &mut evidence.phases)?;
    if evidence.phases.is_empty() {
        evidence
            .warnings
            .push("CUDA-only analysis: expected NVTX phases absent".into());
    }
    Ok(evidence)
}

fn normalize(input: impl BufRead) -> Result<(String, String, Evidence), String> {
    let mut records = Vec::new();
    let mut started = false;
    let mut ended = false;
    for line in input.lines() {
        let event: Event =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if ended {
            return Err("CUPTI records after footer".into());
        }
        match &event {
            Event::Start {
                schema_version: 1,
                api_version: 130301,
                ..
            } if !started => started = true,
            Event::Start { .. } => return Err("unsupported or duplicate CUPTI header".into()),
            _ if !started => return Err("CUPTI header missing".into()),
            Event::End {
                dropped: 0,
                errors: 0,
            } => ended = true,
            Event::End { dropped, errors } => {
                return Err(format!(
                    "CUPTI incomplete: {dropped} dropped, {errors} errors"
                ))
            }
            Event::Error { message } => return Err(format!("CUPTI: {message}")),
            _ => records.push(event),
        }
    }
    if !started || !ended {
        return Err("CUPTI capture has no successful footer".into());
    }
    let origin = records
        .iter()
        .filter_map(|e| match e {
            Event::Kernel { start, .. } | Event::Memop { start, .. } => Some(*start),
            Event::Marker { timestamp, .. } => Some(*timestamp),
            _ => None,
        })
        .min()
        .ok_or("empty CUPTI trace")?;
    let mut open = BTreeMap::new();
    let mut windows = Vec::new();
    let mut ranges = "Name,Start (ns),Duration (ns),PID,TID\n".to_string();
    let mut markers: Vec<_> = records
        .iter()
        .filter(|e| matches!(e, Event::Marker { .. }))
        .collect();
    markers.sort_by_key(|e| match e {
        Event::Marker { timestamp, .. } => *timestamp,
        _ => 0,
    });
    for record in markers {
        let Event::Marker {
            name,
            domain,
            timestamp,
            id,
            flags,
            pid,
            tid,
        } = record
        else {
            continue;
        };
        let key = (*pid, *id);
        if flags & 2 != 0
            && domain.is_empty()
            && crate::nsys::phase_name(name).is_some()
            && open.insert(key, (name.clone(), *timestamp, *tid)).is_some()
        {
            return Err("duplicate CUPTI range start".into());
        }
        if flags & 4 != 0 {
            if let Some((name, start, start_tid)) = open.remove(&key) {
                if *timestamp <= start || *tid != start_tid {
                    return Err("invalid CUPTI local range".into());
                }
                ranges.push_str(&format!(
                    "{},{},{},{},{}\n",
                    csv::field(&name),
                    start - origin,
                    timestamp - start,
                    pid,
                    tid
                ));
                windows.push((name, start, *timestamp));
            }
        }
    }
    if !open.is_empty() {
        return Err("unterminated CUPTI phase range".into());
    }
    let mut gpu = "Name,Start (ns),Duration (ns),Device,Ctx,Strm,Bytes (B),GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (B),DymSMem (B)\n".to_string();
    let mut grouped: BTreeMap<(String, String), Kernel> = BTreeMap::new();
    for record in records {
        match record {
            Event::Kernel {
                name,
                start,
                end,
                device,
                context,
                stream,
                grid,
                block,
                registers,
                shared_bytes,
                ..
            } => {
                if end <= start {
                    return Err("invalid CUPTI kernel duration".into());
                }
                gpu.push_str(&format!(
                    "{},{},{},{},{},{},,{},{},{},{},{},{},{},{},0\n",
                    csv::field(&name),
                    start - origin,
                    end - start,
                    device,
                    context,
                    stream,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    registers,
                    shared_bytes
                ));
                let phase = windows
                    .iter()
                    .find(|(_, a, b)| start >= *a && end <= *b)
                    .map(|w| w.0.as_str())
                    .unwrap_or("");
                for scope in [Some(""), (!phase.is_empty()).then_some(phase)]
                    .into_iter()
                    .flatten()
                {
                    let k = grouped
                        .entry((scope.into(), name.clone()))
                        .or_insert_with(|| Kernel {
                            name: name.clone(),
                            total_ns: 0.0,
                            count: Some(0),
                        });
                    k.total_ns += (end - start) as f64;
                    k.count = k.count.and_then(|n| n.checked_add(1));
                }
            }
            Event::Memop {
                operation,
                start,
                end,
                bytes,
                device,
                context,
                stream,
            } => {
                if end <= start {
                    return Err("invalid CUPTI memory duration".into());
                }
                gpu.push_str(&format!(
                    "{},{},{},{},{},{},{},,,,,,,,,\n",
                    csv::field(&operation),
                    start - origin,
                    end - start,
                    device,
                    context,
                    stream,
                    bytes
                ));
            }
            _ => {}
        }
    }
    let mut evidence = Evidence::default();
    evidence.warnings.push("CUPTI phases use GPU execution contained in the Rust NVTX time window; this is not CUDA API correlation attribution".into());
    for ((phase, _), kernel) in grouped {
        if phase.is_empty() {
            evidence.global.push(kernel);
            continue;
        }
        evidence
            .phases
            .entry(phase.clone())
            .or_insert_with(|| Phase {
                name: phase,
                ..Default::default()
            })
            .kernels
            .push(kernel);
    }
    for kernels in std::iter::once(&mut evidence.global)
        .chain(evidence.phases.values_mut().map(|p| &mut p.kernels))
    {
        kernels.sort_by(|a, b| b.total_ns.total_cmp(&a.total_ns));
    }
    Ok((ranges, gpu, evidence))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_incomplete_activity() {
        for text in [
            "",
            "{\"kind\":\"start\",\"schema_version\":1,\"api_version\":130301,\"pid\":1}\n",
            "{\"kind\":\"end\",\"dropped\":0,\"errors\":0}\n",
        ] {
            assert!(normalize(text.as_bytes()).is_err());
        }
    }
    #[test]
    fn preserves_epoch_precision() {
        let epoch = 1_788_000_000_000_000_000;
        let events = [
            Event::Start {
                schema_version: 1,
                api_version: 130301,
                pid: 1,
            },
            Event::Marker {
                name: "ds4.prefill".into(),
                domain: String::new(),
                timestamp: epoch,
                id: 2,
                flags: 2,
                pid: 1,
                tid: 3,
            },
            Event::Kernel {
                name: "copy".into(),
                start: epoch + 13,
                end: epoch + 113,
                device: 0,
                context: 1,
                stream: 1,
                correlation: 1,
                grid: [2, 1, 1],
                block: [32, 1, 1],
                registers: 16,
                shared_bytes: 0,
                graph_id: 0,
            },
            Event::Marker {
                name: String::new(),
                domain: String::new(),
                timestamp: epoch + 201,
                id: 2,
                flags: 4,
                pid: 1,
                tid: 3,
            },
            Event::End {
                dropped: 0,
                errors: 0,
            },
        ];
        let text = events
            .iter()
            .map(|e| serde_json::to_string(e).unwrap() + "\n")
            .collect::<String>();
        let (ranges, gpu, e) = normalize(text.as_bytes()).unwrap();
        assert!(ranges.contains("ds4.prefill,0,201,1,3"));
        assert!(gpu.contains("copy,13,100,0,1,1"));
        assert_eq!(e.phases["ds4.prefill"].kernels[0].total_ns, 100.0);
        let mut reordered: Vec<_> = events.iter().collect();
        reordered.swap(1, 3);
        let unordered = reordered
            .iter()
            .map(|e| serde_json::to_string(e).unwrap() + "\n")
            .collect::<String>();
        let (paired, _, _) = normalize(unordered.as_bytes()).unwrap();
        assert_eq!(paired, ranges);
        assert!(normalize(text.replace("\"dropped\":0", "\"dropped\":1").as_bytes()).is_err());
    }
}
