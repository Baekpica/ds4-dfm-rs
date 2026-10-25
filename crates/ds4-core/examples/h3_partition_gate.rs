//! Inventory both checkpoint partitions and report which one a request reads.
//!
//! Usage: h3_partition_gate <model_dir> [--references N]
//!
//! Output, in order:
//!
//! - One `COMPONENT` line per component: the four FL2VA directories
//!   (`h3.c:430-433`) and, when installed, the five Ref2VA ones
//!   (`h3.c:927-936`, which is every component, not only the transformer).
//!   The counters are `h3_st_inventory_dir`'s (`h3_safetensors.c:540-585`), so
//!   `tools/minimax_h3/st_probe.c` prints the same four numbers on the
//!   reference side. `table` is an FNV over every tensor's shard, name,
//!   dtype, shape, byte span and offset; `sample` is an FNV over the first
//!   4096 bytes of every tensor. The sample is a fingerprint that tells the
//!   two partitions' weights apart; it is not a whole-file checksum.
//! - `PARTITION` with the partition `--references` selects
//!   (`h3.c:872-875`), or `REFUSED` with the reference's message and exit 3.
//! - `OPEN` with the seconds `h3_create`'s checkpoint check takes.
//!
//! The gate reads shard headers and tensor samples only: no device, no
//! weights resident.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use ds4_core::minimax_h3::partition::{
    Catalog, Partition, AUDIO_VAE, COMPONENTS, FL2VA, REF2VA, TEXT_ENCODER, TOKENIZER, TRANSFORMER,
    VIDEO_VAE,
};
use ds4_core::minimax_h3::safetensors::{open_dir, Shard};

/// Bytes taken from the start of each tensor for the sample fingerprint.
const SAMPLE_BYTES: u64 = 4096;

/// FNV-1a 64-bit, so the fingerprint does not depend on the std hasher.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut root: Option<PathBuf> = None;
    let mut references = 0usize;

    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--references" => {
                index += 1;
                match arguments.get(index).and_then(|value| value.parse().ok()) {
                    Some(value) => references = value,
                    None => return usage("--references needs a count"),
                }
            }
            "--help" | "-h" => return usage(""),
            value if root.is_none() => root = Some(PathBuf::from(value)),
            value => return usage(&format!("unexpected argument {value}")),
        }
        index += 1;
    }

    let Some(root) = root else {
        return usage("a model directory is required");
    };

    let started = Instant::now();
    let catalog = match Catalog::open(&root) {
        Ok(catalog) => catalog,
        Err(error) => {
            eprintln!("{}", error.message());
            return ExitCode::from(3);
        }
    };
    let open_seconds = started.elapsed().as_secs_f64();

    // FL2VA first: it is the baseline every request is allowed to read.
    for (slot, relative) in COMPONENTS.iter().enumerate() {
        let path = catalog.component(Partition::Fl2va, relative);
        let totals = catalog.fl2va()[slot];
        if let Err(detail) = report("fl2va", relative, &path, totals.files) {
            eprintln!("{detail}");
            return ExitCode::FAILURE;
        }
    }
    report_file(
        "fl2va",
        TOKENIZER,
        &catalog.component(Partition::Fl2va, TOKENIZER),
    );

    match catalog.partition(references) {
        Ok(partition) => {
            println!(
                "PARTITION\treferences={references}\tselected={}",
                partition.directory()
            );
        }
        Err(error) => {
            // `h3.c:873-874`: the message is the reference's own.
            println!("REFUSED\treferences={references}\t{}", error.message());
            return ExitCode::from(3);
        }
    }

    if catalog.ref2va_present() {
        let installed = catalog.ref2va_transformer().expect("present");
        let path = catalog.component(Partition::Ref2va, TRANSFORMER);
        if let Err(detail) = report("ref2va", TRANSFORMER, &path, installed.files) {
            eprintln!("{detail}");
            return ExitCode::FAILURE;
        }
        for relative in [TEXT_ENCODER, VIDEO_VAE, AUDIO_VAE] {
            let path = catalog.component(Partition::Ref2va, relative);
            if let Err(detail) = report("ref2va", relative, &path, 0) {
                eprintln!("{detail}");
                return ExitCode::FAILURE;
            }
        }
        report_file(
            "ref2va",
            TOKENIZER,
            &catalog.component(Partition::Ref2va, TOKENIZER),
        );
    } else {
        println!("REF2VA\tpresent=no\tFL2VA alone serves prompt-only requests");
    }

    println!("OPEN\tseconds={open_seconds:.3}");
    ExitCode::SUCCESS
}

/// One file component (`tokenizer.json`), which has no shards.
fn report_file(partition: &str, relative: &str, path: &Path) {
    let bytes = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    println!("COMPONENT\t{partition}/{relative}\tfile_bytes={bytes}");
}

/// Print one shard component's totals and fingerprints. `expected_files` is the
/// catalog's shard count, or zero when the caller has no prior count.
fn report(partition: &str, relative: &str, path: &Path, expected_files: u64) -> Result<(), String> {
    let store = open_dir(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut tensors = 0u64;
    let mut tensor_bytes = 0u64;
    let mut file_bytes = 0u64;
    let mut table = FNV_OFFSET;
    let mut sample = FNV_OFFSET;

    for shard in store.shards() {
        let name = shard_name(shard);
        file_bytes += shard.file_size;
        tensors += shard.tensors.len() as u64;
        tensor_bytes += shard.tensor_bytes();

        let file = File::open(&shard.path)
            .map_err(|error| format!("{}: {error}", shard.path.display()))?;
        for tensor in &shard.tensors {
            let line = format!(
                "{name}\t{}\t{}\t{}\t{}\t{}\n",
                tensor.name,
                tensor.dtype.name(),
                tensor
                    .shape
                    .iter()
                    .map(|axis| axis.to_string())
                    .collect::<Vec<_>>()
                    .join("x"),
                tensor.bytes(),
                tensor.file_offset
            );
            fnv(&mut table, line.as_bytes());

            let span = tensor.bytes().min(SAMPLE_BYTES) as usize;
            let mut buffer = vec![0u8; span];
            file.read_exact_at(&mut buffer, tensor.file_offset)
                .map_err(|error| format!("{}: {error}", shard.path.display()))?;
            fnv(&mut sample, &buffer);
        }
    }

    if expected_files != 0 && expected_files != store.shards().len() as u64 {
        return Err(format!(
            "{partition}/{relative}: the catalog counted {expected_files} shards, the reader {}",
            store.shards().len()
        ));
    }

    println!(
        "COMPONENT\t{partition}/{relative}\tfiles={}\tfile_bytes={file_bytes}\ttensors={tensors}\t\
         tensor_bytes={tensor_bytes}\ttable={table:016x}\tsample={sample:016x}",
        store.shards().len()
    );
    Ok(())
}

fn shard_name(shard: &Shard) -> String {
    shard
        .path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn usage(detail: &str) -> ExitCode {
    if !detail.is_empty() {
        eprintln!("{detail}");
    }
    eprintln!(
        "usage: h3_partition_gate <model_dir> [--references N]\n\
         reports the FL2VA components, the optional Ref2VA components and the partition a\n\
         request reads ({FL2VA} or {REF2VA}); exits 3 when references are refused"
    );
    ExitCode::from(2)
}
