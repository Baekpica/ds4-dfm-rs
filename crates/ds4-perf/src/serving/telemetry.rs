//! Clock queries run beside the workload; every sample keeps its raw stdout.
use super::*;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant, SystemTime},
};

const POLL_MS: u64 = 500;
// Includes query and scheduling delay; larger gaps cannot qualify a run.
const MAX_GAP_MS: u64 = 3 * POLL_MS;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Window {
    poll_interval_ms: u64,
    started_unix_ms: u64,
    finished_unix_ms: u64,
    samples: Vec<context::GpuSample>,
}
impl Window {
    pub(super) fn failures(&self, range: Option<&context::ClockRange>) -> Vec<String> {
        if self.samples.is_empty() {
            return vec!["no in-workload GPU telemetry".into()];
        }
        let mut failures: Vec<_> = self
            .samples
            .iter()
            .flat_map(|sample| sample.failures(range))
            .collect();
        if self.started_unix_ms == 0
            || self.finished_unix_ms < self.started_unix_ms
            || self.samples.iter().any(|sample| {
                let time = sample.timestamp_ms();
                time < self.started_unix_ms.saturating_sub(MAX_GAP_MS)
                    || time > self.finished_unix_ms.saturating_add(MAX_GAP_MS)
            })
            || self.samples.windows(2).any(|pair| {
                let first = pair[0].timestamp_ms();
                let next = pair[1].timestamp_ms();
                next <= first || next - first > MAX_GAP_MS
            })
            || self.samples.first().is_some_and(|sample| {
                sample.timestamp_ms() > self.started_unix_ms.saturating_add(MAX_GAP_MS)
            })
            || self.samples.last().is_some_and(|sample| {
                sample.timestamp_ms().saturating_add(MAX_GAP_MS) < self.finished_unix_ms
            })
        {
            failures.push("GPU telemetry chronology or workload coverage is incomplete".into());
        }
        failures
    }
    pub(super) fn review(
        &self,
        run: &review::Run,
        root: &Path,
        range: Option<&context::ClockRange>,
        expected_device: &Value,
        minimum_ms: f64,
    ) -> Result<(), String> {
        let raw: Value = run.json(&root.join("gpu-window.json"))?;
        if raw != serde_json::to_value(self).map_err(|e| e.to_string())?
            || self.poll_interval_ms != POLL_MS
            || !self.failures(range).is_empty()
            || !minimum_ms.is_finite()
            || minimum_ms < 0.0
            || self.finished_unix_ms.saturating_sub(self.started_unix_ms) as f64 + 1.0 < minimum_ms
        {
            return Err("in-workload GPU telemetry fails contract".into());
        }
        let before: context::GpuSample = run.json(&root.join("gpu-before.json"))?;
        let after: context::GpuSample = run.json(&root.join("gpu-after.json"))?;
        if before.timestamp_ms() > self.started_unix_ms
            || after.timestamp_ms() < self.finished_unix_ms
            || self.samples.iter().any(|sample| {
                sample.timestamp_ms() < before.timestamp_ms()
                    || sample.timestamp_ms() > after.timestamp_ms()
            })
        {
            return Err("GPU polling window lies outside its boundary samples".into());
        }
        for (index, sample) in self.samples.iter().enumerate() {
            let label = format!("gpu-poll-{index:05}");
            let value = serde_json::to_value(sample).map_err(|e| e.to_string())?;
            for key in ["uuid", "name", "driver", "device_ordinal"] {
                if value[key] != expected_device[key] {
                    return Err("GPU changed during workload".into());
                }
            }
            let raw: Value = run.json(&root.join(format!("{label}.json")))?;
            if value != raw {
                return Err("in-workload GPU sample summary changed".into());
            }
            sample.verify_raw(&run.bytes(&root.join(format!("{label}.stdout")))?)?;
        }
        Ok(())
    }
}

pub(super) fn observe<T>(
    args: &cli::Serving,
    dir: &Path,
    action: impl FnOnce() -> Result<T, String>,
) -> Result<(T, Window), String> {
    let finished = AtomicBool::new(false);
    let started_unix_ms = unix_ms()?;
    let (result, samples, finished_unix_ms) = std::thread::scope(|scope| {
        let observer = scope.spawn(|| {
            let mut samples = Vec::new();
            loop {
                let start = Instant::now();
                let label = format!("gpu-poll-{:05}", samples.len());
                samples.push(context::gpu(args, dir, &label)?);
                while !finished.load(Ordering::Acquire)
                    && start.elapsed() < Duration::from_millis(POLL_MS)
                {
                    std::thread::sleep(Duration::from_millis(10));
                }
                if finished.load(Ordering::Acquire) {
                    return Ok::<_, String>(samples);
                }
            }
        });
        let result = action();
        let finished_unix_ms = unix_ms();
        finished.store(true, Ordering::Release);
        (
            result,
            observer.join().map_err(|_| "GPU observer panicked"),
            finished_unix_ms,
        )
    });
    let window = Window {
        poll_interval_ms: POLL_MS,
        started_unix_ms,
        finished_unix_ms: finished_unix_ms?,
        samples: samples??,
    };
    fs::write(
        dir.join("gpu-window.json"),
        serde_json::to_vec_pretty(&window).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok((result?, window))
}

fn unix_ms() -> Result<u64, String> {
    Ok(SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn intermediate_clock_spike_fails_contract() {
        let range: context::ClockRange =
            serde_json::from_value(json!({"min":300,"max":2200})).unwrap();
        let sample = |clock, time| json!({"label":"gpu-poll","unix_ms":time,"device_ordinal":0,"uuid":"fixture","name":"fixture","driver":"fixture","sm_clock_mhz":clock,"temperature_c":50,"unavailable_reason":null});
        let window: Window = serde_json::from_value(
            json!({"poll_interval_ms":500,"started_unix_ms":1,"finished_unix_ms":1001,"samples":[sample(1000,1),sample(2500,501),sample(1000,1001)]}),
        )
        .unwrap();
        assert_eq!(window.failures(Some(&range)).len(), 1);
    }

    #[test]
    fn gpu_window_needs_coverage() {
        for (times, complete) in [
            (vec![1000, 900], false),
            (vec![1000], false),
            (vec![1000, 5000], false),
            (vec![9000], false),
            (vec![1000, 2000, 3000, 4000, 5000], true),
        ] {
            let samples: Vec<_> = times
                .iter()
                .map(|time| {
                    json!({
                        "label":"gpu-poll","unix_ms":time,"device_ordinal":0,"uuid":"fixture",
                        "name":"fixture","driver":"fixture","sm_clock_mhz":1000,
                        "temperature_c":50,"unavailable_reason":null
                    })
                })
                .collect();
            let window: Window = serde_json::from_value(json!({
                "poll_interval_ms":500,"started_unix_ms":1000,"finished_unix_ms":5000,
                "samples":samples
            }))
            .unwrap();
            assert_eq!(
                window.failures(None).is_empty(),
                complete,
                "GPU samples: {times:?}"
            );
        }
    }
}
