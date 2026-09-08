use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gpu {
    pub ordinal: usize,
    pub name: String,
    pub uuid: String,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub driver_version: i32,
    pub total_memory_bytes: u64,
    pub free_memory_bytes: u64,
    pub multiprocessors: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_sm: i32,
    pub max_blocks_per_sm: i32,
    pub registers_per_sm: i32,
    pub shared_bytes_per_sm: i32,
    pub shared_bytes_per_block: i32,
    pub l2_bytes: i32,
    pub memory_bus_bits: Option<i32>,
    pub memory_clock_khz: Option<i32>,
    pub clock_khz: Option<i32>,
    pub unavailable: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Machine {
    pub facts: BTreeMap<String, String>,
    pub gpu: Option<Gpu>,
    pub gpu_processes: String,
    pub process_query_ok: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    pub gpu: Gpu,
    pub transfer_bytes: u64,
    pub warmups: u32,
    pub repeats: u32,
    pub copy_gb_s: Vec<f64>,
    pub fp32_gflop_s: Vec<f64>,
    pub launch_us: Vec<f64>,
    pub method: String,
    pub source_sha256: String,
    pub ptx_sha256: String,
    pub copy_ms: Vec<f64>,
    pub compute_ms: Vec<f64>,
    pub launch_batch_ms: Vec<f64>,
    pub compute_threads: u32,
    pub compute_block_threads: u32,
    pub fma_iterations: u32,
    pub launch_count: u32,
    pub notes: Vec<String>,
}

impl crate::artifact::Payload for Machine {
    const KIND: &'static str = "machine";
    fn validate(&self) -> Result<(), String> {
        if let Some(gpu) = &self.gpu {
            gpu.validate()?;
        }
        Ok(())
    }
}

impl Gpu {
    pub fn validate(&self) -> Result<(), String> {
        if self.uuid.is_empty()
            || self.name.is_empty()
            || self.multiprocessors <= 0
            || self.warp_size <= 0
            || self.max_threads_per_sm <= 0
            || self.max_threads_per_block <= 0
            || self.max_blocks_per_sm <= 0
            || self.total_memory_bytes == 0
            || self.free_memory_bytes > self.total_memory_bytes
        {
            return Err("invalid CUDA device properties".into());
        }
        Ok(())
    }
    /// Free memory is a dynamic observation, excluded from comparison identity.
    pub fn same_device(&self, other: &Self) -> bool {
        self.uuid == other.uuid
            && self.driver_version == other.driver_version
            && self.compute_major == other.compute_major
            && self.compute_minor == other.compute_minor
    }
}

impl crate::artifact::Payload for Calibration {
    const KIND: &'static str = "calibration";
    fn validate(&self) -> Result<(), String> {
        self.gpu.validate()?;
        if self.method != "cuda-event-copy-fp32-fma-v1"
            || self.source_sha256.len() != 64
            || self.ptx_sha256.len() != 64
            || self.repeats == 0
            || self.repeats > 100
            || self.transfer_bytes <= self.gpu.l2_bytes.max(0) as u64
            || self.compute_threads == 0
            || self.compute_block_threads == 0
            || self.fma_iterations == 0
            || self.launch_count == 0
        {
            return Err("invalid calibration method or workload".into());
        }
        for values in [
            &self.copy_gb_s,
            &self.fp32_gflop_s,
            &self.launch_us,
            &self.copy_ms,
            &self.compute_ms,
            &self.launch_batch_ms,
        ] {
            if values.len() != self.repeats as usize
                || values.iter().any(|v| !v.is_finite() || *v <= 0.0)
            {
                return Err("invalid calibration samples".into());
            }
        }
        for i in 0..self.repeats as usize {
            let calculated = [
                2.0 * self.transfer_bytes as f64 / (self.copy_ms[i] * 1e6),
                self.compute_threads as f64 * self.fma_iterations as f64 * 16.0
                    / (self.compute_ms[i] * 1e6),
                self.launch_batch_ms[i] * 1000.0 / self.launch_count as f64,
            ];
            let reported = [self.copy_gb_s[i], self.fp32_gflop_s[i], self.launch_us[i]];
            if calculated
                .iter()
                .zip(reported)
                .any(|(a, b)| (a - b).abs() > a.abs() * 1e-9)
            {
                return Err("calibration rates disagree with raw timings".into());
            }
        }
        Ok(())
    }
}
