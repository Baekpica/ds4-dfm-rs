//! Inventory a MiniMax-H3 safetensors directory, optionally reconciled
//! against its HF index.
//!
//! Usage: minimax_h3_inspect <dir> [index.json]
//!        minimax_h3_inspect --layout <dir>
//!
//! Output is one TSV row per tensor (shard, name, dtype, dims, bytes, offset),
//! then a TOTAL line, then the manifest when an index is given. The reference
//! engine's reader prints the same shape (`tools/minimax_h3/st_probe.c` in the
//! port evidence), which is how P3's two-sided diff is produced.
//!
//! `--layout` checks the shards against `config.json` instead: one TSV row per
//! tensor (name, role, dtype, dims), then a LAYOUT line. A mismatch prints its
//! token and exits non-zero.

use std::path::Path;
use std::process::ExitCode;

use ds4_core::minimax_h3::layout::{Layout, Role};
use ds4_core::minimax_h3::safetensors::{open_dir, reconcile};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        eprintln!("usage: minimax_h3_inspect <dir> [index.json]");
        return ExitCode::from(2);
    }

    let (layout, args) = match args.split_first() {
        Some((flag, rest)) if flag == "--layout" => (true, rest),
        _ => (false, args.as_slice()),
    };

    if args.is_empty() {
        eprintln!("usage: minimax_h3_inspect [--layout] <dir> [index.json]");
        return ExitCode::from(2);
    }

    let dir = Path::new(&args[0]);

    if layout {
        return report_layout(dir);
    }

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

/// Check the shards against the config and print the checked roles.
fn report_layout(dir: &Path) -> ExitCode {
    let layout = match Layout::open(dir) {
        Ok(layout) => layout,
        Err(error) => {
            eprintln!("{}", error.token());
            return ExitCode::FAILURE;
        }
    };

    for entry in layout.entries() {
        let dims = entry
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join("x");

        println!(
            "{}\t{}\t{}\t{dims}",
            entry.name,
            role_name(entry.role),
            entry.dtype.name()
        );
    }

    let config = layout.config();
    println!(
        "LAYOUT\ttensors={}\tblocks={}\trefiner={}\thidden={}\tinner={}\tvideo_patch={}",
        layout.entries().len(),
        config.layers,
        config.refiner_layers,
        config.hidden,
        config.inner(),
        config.video_patch()
    );

    ExitCode::SUCCESS
}

/// The role with its block index, for the TSV column.
fn role_name(role: Role) -> String {
    match role {
        Role::BlockNorm1(block) => format!("block_norm1[{block}]"),
        Role::BlockNorm2(block) => format!("block_norm2[{block}]"),
        Role::BlockQkv(block) => format!("block_qkv[{block}]"),
        Role::BlockQNorm(block) => format!("block_q_norm[{block}]"),
        Role::BlockKNorm(block) => format!("block_k_norm[{block}]"),
        Role::BlockOut(block) => format!("block_out[{block}]"),
        Role::BlockFc1(block) => format!("block_fc1[{block}]"),
        Role::BlockFc2(block) => format!("block_fc2[{block}]"),
        Role::BlockAdalnWeight(block) => format!("block_adaln_weight[{block}]"),
        Role::BlockAdalnBias(block) => format!("block_adaln_bias[{block}]"),
        Role::RefinerNorm1(block) => format!("refiner_norm1[{block}]"),
        Role::RefinerNorm2(block) => format!("refiner_norm2[{block}]"),
        Role::RefinerQkv(block) => format!("refiner_qkv[{block}]"),
        Role::RefinerQNorm(block) => format!("refiner_q_norm[{block}]"),
        Role::RefinerKNorm(block) => format!("refiner_k_norm[{block}]"),
        Role::RefinerOut(block) => format!("refiner_out[{block}]"),
        Role::RefinerFc1(block) => format!("refiner_fc1[{block}]"),
        Role::RefinerFc2(block) => format!("refiner_fc2[{block}]"),
        other => format!("{other:?}"),
    }
}
