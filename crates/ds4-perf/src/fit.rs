use crate::{csv, nsys};
use ds4_perf::{
    artifact::{self, Artifact, Payload},
    machine::{Calibration, Gpu, Machine},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    pub phase: String,
    pub kernel: String,
    pub grid: [u64; 3],
    pub block: [u64; 3],
    pub registers_per_thread: u64,
    pub shared_bytes: u64,
    pub instances: u64,
    pub total_ns: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bound {
    pub operand_shape: Option<BTreeMap<String, u64>>,
    pub operand_shape_source: String,
    pub launch: Launch,
    pub resident_blocks_upper: u64,
    pub occupancy_upper: f64,
    pub waves_lower: u64,
    pub last_wave_fill: f64,
    pub block_limits: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fit {
    pub calibration_status: String,
    pub workload_shape_status: String,
    pub bounds: Vec<Bound>,
    pub unknown_launches: u64,
    pub observed_copy_gb_s: Option<f64>,
    pub observed_fp32_gflop_s: Option<f64>,
    pub observed_launch_us: Option<f64>,
    pub shape: BTreeMap<String, u64>,
    pub limits: Vec<String>,
}

impl Payload for Fit {
    const KIND: &'static str = "fit";
    fn validate(&self) -> Result<(), String> {
        for b in &self.bounds {
            if b.resident_blocks_upper == 0
                || b.waves_lower == 0
                || !b.occupancy_upper.is_finite()
                || !(0.0..=1.0).contains(&b.occupancy_upper)
                || !b.last_wave_fill.is_finite()
                || !(0.0..=1.0).contains(&b.last_wave_fill)
                || !b.launch.total_ns.is_finite()
                || b.launch.total_ns <= 0.0
            {
                return Err("invalid geometry bound".into());
            }
        }
        for v in [
            self.observed_copy_gb_s,
            self.observed_fp32_gflop_s,
            self.observed_launch_us,
        ]
        .into_iter()
        .flatten()
        {
            if !v.is_finite() || v <= 0.0 {
                return Err("invalid measured envelope".into());
            }
        }
        Ok(())
    }
}

fn product(values: [u64; 3]) -> Result<u64, String> {
    values.into_iter().try_fold(1u64, |a, b| {
        a.checked_mul(b)
            .filter(|n| *n > 0)
            .ok_or("invalid launch dimensions".into())
    })
}

fn bound(gpu: &Gpu, launch: Launch) -> Result<Bound, String> {
    gpu.validate()?;
    let threads = product(launch.block)?;
    let blocks = product(launch.grid)?;
    if threads > gpu.max_threads_per_block as u64 || launch.registers_per_thread == 0 {
        return Err("invalid block size or unknown register count".into());
    }
    let warp_threads = threads.div_ceil(gpu.warp_size as u64) * gpu.warp_size as u64;
    let registers = warp_threads
        .checked_mul(launch.registers_per_thread)
        .ok_or("register demand overflow")?;
    let mut limits = BTreeMap::from([
        ("block_slots".into(), gpu.max_blocks_per_sm as u64),
        (
            "threads".into(),
            gpu.max_threads_per_sm as u64 / warp_threads,
        ),
        (
            "registers".into(),
            gpu.registers_per_sm.max(0) as u64 / registers,
        ),
    ]);
    if let Some(blocks) = (gpu.shared_bytes_per_sm.max(0) as u64).checked_div(launch.shared_bytes) {
        limits.insert("shared_memory".into(), blocks);
    }
    let resident = *limits.values().min().ok_or("resource limits unavailable")?;
    if resident == 0 {
        return Err("launch exceeds reported SM resources".into());
    }
    let capacity = resident
        .checked_mul(gpu.multiprocessors as u64)
        .ok_or("grid capacity overflow")?;
    let waves = blocks.div_ceil(capacity);
    Ok(Bound {
        operand_shape: None,
        operand_shape_source:
            "unavailable: activity records expose launch geometry, not operand dimensions".into(),
        launch,
        resident_blocks_upper: resident,
        occupancy_upper: resident as f64 * warp_threads as f64 / gpu.max_threads_per_sm as f64,
        waves_lower: waves,
        last_wave_fill: (1 + (blocks - 1) % capacity) as f64 / capacity as f64,
        block_limits: limits,
    })
}

fn byte_value(header: &[String], row: &[String], name: &str) -> Option<u64> {
    // Resource bounds require integer bytes. Rounded MB/KB reports cannot
    // support an occupancy upper bound; scout requests csv:noconv.
    csv::value(row, header, &[&format!("{name} (b)")])?
        .parse()
        .ok()
}

fn launches(out: &Path) -> Result<(Vec<Launch>, u64), String> {
    let mut ranges = Vec::new();
    csv::table(
        csv::open(&out.join("nsys-ranges.csv")).map_err(|e| e.to_string())?,
        "name",
        |header, row| {
            let Some(name) = csv::value(row, header, &["name"]).and_then(nsys::phase_name) else {
                return Ok(());
            };
            let start = csv::time(row, header, "Start").ok_or("range Start missing")?;
            let duration = csv::time(row, header, "Duration").ok_or("range Duration missing")?;
            ranges.push((name, start, start + duration));
            Ok(())
        },
    )?;
    type LaunchKey = (String, String, [u64; 3], [u64; 3], u64, u64);
    let mut groups: BTreeMap<LaunchKey, Launch> = BTreeMap::new();
    let mut unknown = 0;
    csv::table(
        csv::open(&out.join("nsys-gpu-trace.csv")).map_err(|e| e.to_string())?,
        "name",
        |header, row| {
            let name = csv::value(row, header, &["name"]).ok_or("kernel Name missing")?;
            if header
                .iter()
                .position(|s| csv::key(s).starts_with("bytes ("))
                .and_then(|i| row.get(i))
                .is_some_and(|s| !s.trim().is_empty())
                || name.starts_with("[CUDA ")
            {
                return Ok(());
            }
            let start = csv::time(row, header, "Start").ok_or("kernel Start missing")?;
            let duration = csv::time(row, header, "Duration").ok_or("kernel Duration missing")?;
            let Some((phase, _, _)) = ranges
                .iter()
                .find(|(_, a, b)| start >= *a && start + duration <= *b)
            else {
                return Ok(());
            };
            let integer =
                |name| csv::value(row, header, &[name]).and_then(|v| v.parse::<u64>().ok());
            let dimensions: Option<Vec<_>> =
                ["grdx", "grdy", "grdz", "blkx", "blky", "blkz", "reg/trd"]
                    .into_iter()
                    .map(integer)
                    .collect();
            let shared = byte_value(header, row, "stcsmem")
                .zip(byte_value(header, row, "dymsmem"))
                .and_then(|(a, b)| a.checked_add(b));
            let (Some(d), Some(shared)) = (dimensions, shared) else {
                unknown += 1;
                return Ok(());
            };
            let grid = [d[0], d[1], d[2]];
            let block = [d[3], d[4], d[5]];
            let key = (
                phase.to_string(),
                name.to_string(),
                grid,
                block,
                d[6],
                shared,
            );
            let launch = groups.entry(key).or_insert_with(|| Launch {
                phase: phase.to_string(),
                kernel: name.to_string(),
                grid,
                block,
                registers_per_thread: d[6],
                shared_bytes: shared,
                instances: 0,
                total_ns: 0.0,
            });
            launch.instances += 1;
            launch.total_ns += duration;
            Ok(())
        },
    )?;
    Ok((groups.into_values().collect(), unknown))
}

pub fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

pub fn run(
    out: &Path,
    machine: &Artifact<Machine>,
    calibration: Option<&Path>,
    shape: BTreeMap<String, u64>,
) -> Result<(), String> {
    let result: Result<Fit, String> = (|| {
        let gpu = machine
            .require()?
            .gpu
            .as_ref()
            .ok_or("fit requires CUDA device properties")?;
        let calibration = calibration.map(artifact::load::<Calibration>).transpose()?;
        let envelope = calibration.as_ref().map(|c| c.require()).transpose()?;
        if envelope.is_some_and(|c| !c.gpu.same_device(gpu)) {
            return Err("calibration device/driver differs from scout".into());
        }
        let (launches, mut unknown) = launches(out)?;
        let mut bounds = Vec::new();
        let mut limits = vec!["Resource occupancy is an upper bound: allocation granularity, carveout, clusters and barriers are not modeled. Waves/fill use that bound, not observed occupancy.".into(),"Operand shapes are declared workload metadata. Kernel names do not establish M/N/K, tensor-core FLOPs, or memory traffic.".into(),"Calibration is an observed FP32 SIMT/copy/host-launch envelope, not a tensor-core peak or a workload roofline.".into()];
        for launch in launches {
            let count = launch.instances;
            match bound(gpu, launch) {
                Ok(b) => bounds.push(b),
                Err(e) => {
                    unknown += count;
                    limits.push(e);
                }
            }
        }
        bounds.sort_by(|a, b| b.launch.total_ns.total_cmp(&a.launch.total_ns));
        Ok(Fit {
            calibration_status: if envelope.is_some() {
                "measured"
            } else {
                "missing"
            }
            .into(),
            workload_shape_status: if shape.is_empty() {
                "missing"
            } else {
                "declared"
            }
            .into(),
            bounds,
            unknown_launches: unknown,
            observed_copy_gb_s: envelope.map(|c| median(&c.copy_gb_s)),
            observed_fp32_gflop_s: envelope.map(|c| median(&c.fp32_gflop_s)),
            observed_launch_us: envelope.map(|c| median(&c.launch_us)),
            shape,
            limits,
        })
    })();
    let mut artifact = match result {
        Ok(data) => {
            let mut a = Artifact::new(data);
            a.complete = a.data.as_ref().is_some_and(|d| {
                !d.bounds.is_empty()
                    && d.unknown_launches == 0
                    && d.calibration_status == "measured"
                    && d.workload_shape_status == "declared"
            });
            a
        }
        Err(ref error) => Artifact::<Fit>::failed(error.clone()),
    };
    for path in [
        Some(out.join("collection.json")),
        Some(out.join("machine/machine.json")),
        calibration.map(Path::to_path_buf),
    ]
    .into_iter()
    .flatten()
    {
        if path.is_file() {
            artifact.inputs.push(artifact::reference(&path, out)?);
        }
    }
    artifact::save(&out.join("fit.json"), &artifact)?;
    if !artifact.complete {
        return Err(
            "fit incomplete; see fit.json for missing geometry, calibration or workload metadata"
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn gpu() -> Gpu {
        serde_json::from_value(serde_json::json!({
            "ordinal":0,"name":"fixture","uuid":"test","compute_major":12,"compute_minor":1,"driver_version":13030,
            "total_memory_bytes":1024,"free_memory_bytes":512,"multiprocessors":2,"warp_size":32,
            "max_threads_per_block":1024,"max_threads_per_sm":1536,"max_blocks_per_sm":24,
            "registers_per_sm":65536,"shared_bytes_per_sm":102400,"shared_bytes_per_block":101376,"l2_bytes":1,
            "memory_bus_bits":null,"memory_clock_khz":null,"clock_khz":null,"unavailable":{}
        })).unwrap()
    }
    fn launch() -> Launch {
        Launch {
            phase: "ds4.prefill".into(),
            kernel: "fixture".into(),
            grid: [49, 1, 1],
            block: [33, 1, 1],
            registers_per_thread: 16,
            shared_bytes: 0,
            instances: 1,
            total_ns: 100.0,
        }
    }
    #[test]
    fn rounds_warps_and_grid_tail() {
        let b = bound(&gpu(), launch()).unwrap();
        assert_eq!(b.resident_blocks_upper, 24);
        assert_eq!(b.occupancy_upper, 1.0);
        assert_eq!(b.waves_lower, 2);
        assert_eq!(b.last_wave_fill, 1.0 / 48.0);
    }
    #[test]
    fn limits_shared_and_unknowns() {
        let mut l = launch();
        l.shared_bytes = 51200;
        assert_eq!(bound(&gpu(), l.clone()).unwrap().resident_blocks_upper, 2);
        l.registers_per_thread = 0;
        assert!(bound(&gpu(), l).is_err());
        assert!(product([u64::MAX, 2, 1]).is_err());
        assert_eq!(
            byte_value(&["StcSMem (MB)".into()], &["0.01".into()], "stcsmem"),
            None
        );
    }
}
