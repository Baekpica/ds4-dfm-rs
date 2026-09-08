use crate::{
    doctor,
    runner::{self, Probe},
};
use ds4_perf::{
    artifact::{self, Artifact, Payload},
    machine::{Calibration, Gpu, Machine},
};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

pub fn helper(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return path.canonicalize().ok();
    }
    let sibling = std::env::current_exe().ok()?.with_file_name("ds4-perf-gpu");
    if sibling.is_file() {
        return Some(sibling);
    }
    runner::resolve("ds4-perf-gpu".as_ref())
}

fn raw_inputs<T>(artifact: &mut Artifact<T>, out: &Path, prefix: &str) -> Result<(), String> {
    for suffix in ["command.txt", "stdout", "stderr", "status.txt"] {
        let path = out.join(format!("{prefix}.{suffix}"));
        if path.is_file() {
            artifact.inputs.push(artifact::reference(&path, out)?);
        }
    }
    Ok(())
}

pub fn collect(
    out: &Path,
    device: usize,
    gpu_helper: Option<&Path>,
    bench: Option<&str>,
) -> Result<Artifact<Machine>, String> {
    let mut system = runner::System;
    let d = doctor::inspect(&mut system, bench);
    d.save_probes(out)?;
    runner::write(&out.join("doctor.txt"), &d.render())?;
    let processes = system.capture(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
    );
    let mut warnings = d.warnings;
    let mut facts = d.facts;
    if let Ok(exe) = std::env::current_exe() {
        facts.insert("ds4_perf_sha256".into(), artifact::hash(&exe)?);
    }
    let gpu = match helper(gpu_helper) {
        Some(path) => {
            let before = artifact::hash(&path)?;
            facts.insert("gpu_helper_path".into(), path.to_string_lossy().into());
            facts.insert("gpu_helper_sha256".into(), before.clone());
            let command = [
                path.clone().into_os_string(),
                "inspect".into(),
                device.to_string().into(),
            ];
            match runner::run(&command, out, "gpu-inspect").and_then(|_| {
                if before != artifact::hash(&path)? {
                    return Err("GPU helper changed during inspection".into());
                }
                let bytes =
                    std::fs::read(out.join("gpu-inspect.stdout")).map_err(|e| e.to_string())?;
                let gpu: Gpu = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                gpu.validate()?;
                if gpu.ordinal != device {
                    return Err("helper inspected another device".into());
                }
                Ok(gpu)
            }) {
                Ok(gpu) => Some(gpu),
                Err(error) => {
                    warnings.push(format!("CUDA device properties unavailable: {error}"));
                    None
                }
            }
        }
        None => {
            warnings.push(
                "CUDA helper unavailable; build make ds4-perf-gpu or pass --gpu-helper".into(),
            );
            None
        }
    };
    if !processes.ok {
        warnings.push("GPU process state unknown".into());
    }
    let complete = gpu.is_some() && processes.ok;
    let mut machine = Artifact::new(Machine {
        facts,
        gpu,
        gpu_processes: processes.combined(),
        process_query_ok: processes.ok,
    });
    machine.complete = complete;
    machine.warnings = warnings;
    raw_inputs(&mut machine, out, "gpu-inspect")?;
    machine
        .inputs
        .push(artifact::reference(&out.join("doctor.txt"), out)?);
    artifact::save(&out.join("machine.json"), &machine)?;
    Ok(machine)
}

pub fn run(
    out: &Path,
    device: usize,
    calibrate: bool,
    gpu_helper: Option<&Path>,
    bench: Option<&str>,
) -> Result<(), String> {
    artifact::directory(out)?;
    let machine = collect(out, device, gpu_helper, bench)?;
    println!(
        "machine: {} ({})",
        out.join("machine.json").display(),
        if machine.complete {
            "device properties measured"
        } else {
            "partial; see warnings"
        }
    );
    if !calibrate {
        return Ok(());
    }
    let result = calibrate_device(out, device, gpu_helper, &machine);
    let mut calibration = match &result {
        Ok(data) => Artifact::new(data.clone()),
        Err(error) => Artifact::<Calibration>::failed(error.clone()),
    };
    calibration
        .inputs
        .push(artifact::reference(&out.join("machine.json"), out)?);
    raw_inputs(&mut calibration, out, "calibration")?;
    artifact::save(&out.join("calibration.json"), &calibration)?;
    result?;
    println!("calibration: {}", out.join("calibration.json").display());
    Ok(())
}

fn calibrate_device(
    out: &Path,
    device: usize,
    gpu_helper: Option<&Path>,
    machine: &Artifact<Machine>,
) -> Result<Calibration, String> {
    let inspected = machine.require()?;
    let helper =
        helper(gpu_helper).ok_or("calibration requires ds4-perf-gpu; build make ds4-perf-gpu")?;
    let hash = artifact::hash(&helper)?;
    if inspected.facts.get("gpu_helper_sha256") != Some(&hash) {
        return Err("GPU helper changed since inspection".into());
    }
    let command: Vec<OsString> = vec![
        helper.clone().into_os_string(),
        "calibrate".into(),
        device.to_string().into(),
    ];
    runner::run(&command, out, "calibration")?;
    if artifact::hash(&helper)? != hash {
        return Err("GPU helper changed during calibration".into());
    }
    let data: Calibration = serde_json::from_slice(
        &std::fs::read(out.join("calibration.stdout")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    data.validate()?;
    if !inspected
        .gpu
        .as_ref()
        .is_some_and(|gpu| gpu.same_device(&data.gpu) && gpu.ordinal == data.gpu.ordinal)
    {
        return Err("calibration device differs from machine inspection".into());
    }
    Ok(data)
}
