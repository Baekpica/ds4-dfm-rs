//! The device-probe gate: the port's record against the reference's checks.
//!
//! Usage: h3_device_gate [--expect-unified yes|no]
//!
//! Prints the DEVICE line `h3-rs --info` prints, then the reference test's own
//! summary line (`tests/test_device.c:22-24`): name, architecture, GPU memory
//! from `recommended_working_set`, and the unified flag. The checks are
//! `tests/test_device.c:13-17` — a non-empty name and architecture and both
//! capacity fields non-zero — plus the one thing the reference leaves implicit:
//! its probe formats the architecture as "CUDA sm_<major><minor>"
//! (`h3_device_cuda.cu:39-40`), so an empty-looking or mismatched string fails.
//!
//! The values are machine facts, not goldens: run the reference's
//! `h3_device_test` beside this gate on the same host and compare the two
//! lines field by field.

use std::process::ExitCode;

use ds4_core::minimax_h3::device;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(detail) => {
            eprintln!("FAIL device probe: {detail}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let mut expect_unified = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--expect-unified" => {
                let value = args.next().ok_or("--expect-unified needs yes or no")?;
                expect_unified = Some(match value.as_str() {
                    "yes" => true,
                    "no" => false,
                    _ => return Err("--expect-unified needs yes or no".to_string()),
                });
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }

    let device = device::probe()?;
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let unified = if device.unified_memory { "yes" } else { "no" };

    println!(
        "DEVICE name=\"{}\" architecture=\"{}\" physical_gib={:.1} working_set_gib={:.1} max_buffer_gib={:.1} unified={}",
        device.name,
        device.architecture,
        gib(device.physical_memory),
        gib(device.recommended_working_set),
        gib(device.max_buffer_length),
        unified
    );

    // tests/test_device.c:13-17.
    if device.name.is_empty() || device.architecture.is_empty() {
        return Err("incomplete device information".to_string());
    }
    if device.recommended_working_set == 0 || device.max_buffer_length == 0 {
        return Err("incomplete device information".to_string());
    }

    // h3_device_cuda.cu:39-40: the reference builds this string itself.
    let sm = device
        .architecture
        .strip_prefix("CUDA sm_")
        .ok_or_else(|| {
            format!(
                "architecture is not the reference's format: {}",
                device.architecture
            )
        })?;
    if sm.is_empty() || !sm.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "architecture is not the reference's format: {}",
            device.architecture
        ));
    }

    if let Some(expected) = expect_unified {
        if expected != device.unified_memory {
            return Err(format!("unified memory is {unified}, expected {expected}"));
        }
    }

    println!(
        "ok: {} ({}), {:.1} GiB GPU memory, unified={}",
        device.name,
        device.architecture,
        gib(device.recommended_working_set),
        unified
    );

    Ok(())
}
