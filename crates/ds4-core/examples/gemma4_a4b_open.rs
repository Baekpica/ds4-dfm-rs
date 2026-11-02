//! Open a `vq-release/1` artifact as the Gemma-4-26B-A4B family and print
//! what bound, without a device.
//!
//!     cargo run -p ds4-core --example gemma4_a4b_open -- <artifact-dir> [--verify]
//!
//! `--verify` re-reads every file's sha256 against the release; the plain
//! open checks each file's size and container header and reads no weights.

use std::path::PathBuf;
use std::process::ExitCode;

use ds4_core::gemma4_a4b::{Family, LayerType, Role};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("usage: gemma4_a4b_open <artifact-dir> [--verify]");
            return ExitCode::from(2);
        }
    };
    let verify = args.any(|a| a == "--verify");
    match run(&path, verify) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gemma4_a4b_open: {error}");
            ExitCode::from(1)
        }
    }
}

fn run(path: &std::path::Path, verify: bool) -> Result<(), String> {
    let family = Family::open(path).map_err(|e| e.to_string())?;
    let config = &family.config;
    println!(
        "family      gemma4_a4b {} layers, hidden {}, vocab {}",
        config.layers, config.hidden, config.vocab
    );
    println!(
        "attention   {} sliding (head {}), {} full (head {})",
        config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::Sliding)
            .count(),
        config.head_dim,
        config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::Full)
            .count(),
        config.global_head_dim
    );
    println!(
        "artifact    {} entries ({} encoded, {} copied), {} bytes",
        family.release.totals.entries,
        family.release.totals.encoded,
        family.release.totals.copied,
        family.release.totals.bytes
    );
    println!("fingerprint {}", family.release.totals.fingerprint);
    for (index, layer) in family.layout.layers.iter().enumerate().take(2) {
        println!(
            "layer {index:<2}   {:?}, {} roles, expert gate-up {}x{}",
            layer.kind,
            layer.roles.len(),
            layer
                .get(Role::ExpertGateUp)
                .map_err(|e| e.to_string())?
                .rows,
            layer
                .get(Role::ExpertGateUp)
                .map_err(|e| e.to_string())?
                .cols
        );
    }
    let full = family
        .layout
        .layers
        .iter()
        .position(|l| l.kind == LayerType::Full)
        .ok_or("no full-attention layer bound")?;
    println!(
        "full layer  {full}: q_proj {}x{}, k_proj {}x{}, v_proj {}",
        family
            .role(full, Role::QProj)
            .map_err(|e| e.to_string())?
            .rows,
        family
            .role(full, Role::QProj)
            .map_err(|e| e.to_string())?
            .cols,
        family
            .role(full, Role::KProj)
            .map_err(|e| e.to_string())?
            .rows,
        family
            .role(full, Role::KProj)
            .map_err(|e| e.to_string())?
            .cols,
        if family.role(full, Role::VProj).is_ok() {
            "present"
        } else {
            "absent (k_eq_v)"
        }
    );
    if verify {
        family.verify().map_err(|e| e.to_string())?;
        println!("verify      every sha256 matches");
    }
    Ok(())
}
