//! Inventory a MiniMax-H3 safetensors directory, optionally reconciled
//! against its HF index.
//!
//! Usage: h3_inspect <dir> [index.json]
//!
//! Output is one TSV row per tensor (shard, name, dtype, dims, bytes, offset),
//! then a TOTAL line, then the manifest when an index is given. The reference
//! engine's reader prints the same shape (`tools/h3_st_probe.c` in the port
//! evidence), which is how P3's two-sided diff is produced.

use std::path::Path;
use std::process::ExitCode;

use ds4_core::minimax_h3::safetensors::{open_dir, reconcile};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        eprintln!("usage: h3_inspect <dir> [index.json]");
        return ExitCode::from(2);
    }

    let dir = Path::new(&args[0]);

    let store = match open_dir(dir) {
        Ok(store) => store,
        Err(error) => {
            eprintln!("{}", error.token());
            return ExitCode::FAILURE;
        }
    };

    let mut rows = Vec::new();
    for shard in store.shards() {
        let file = shard
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();

        for tensor in &shard.tensors {
            let dims = tensor
                .shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join("x");

            rows.push((
                file.clone(),
                tensor.name.clone(),
                tensor.dtype.name().to_string(),
                dims,
                tensor.bytes(),
                tensor.file_offset,
            ));
        }
    }

    rows.sort();

    for (file, name, dtype, dims, bytes, offset) in &rows {
        println!("{file}\t{name}\t{dtype}\t{dims}\t{bytes}\t{offset}");
    }

    let file_bytes: u64 = store.shards().iter().map(|shard| shard.file_size).sum();
    println!(
        "TOTAL\tshards={}\ttensors={}\ttensor_bytes={}\tfile_bytes={}",
        store.shards().len(),
        store.tensor_count(),
        store.tensor_bytes(),
        file_bytes
    );

    if let Some(index) = args.get(1) {
        match reconcile(dir, Path::new(index)) {
            Ok(manifest) => println!(
                "MANIFEST\tshards={}\ttensors={}\ttensor_bytes={}",
                manifest.shards, manifest.tensors, manifest.tensor_bytes
            ),
            Err(error) => {
                eprintln!("{}", error.token());
                return ExitCode::FAILURE;
            }
        }
    }

    ExitCode::SUCCESS
}
