//! Inventory a MiniMax-H3 safetensors directory, optionally reconciled
//! against its HF index.
//!
//! Usage: minimax_h3_inspect <dir> [index.json]
//!        minimax_h3_inspect --layout <dir>
//!        minimax_h3_inspect --weights <dir>
//!        minimax_h3_inspect --text <FL2VA/text_encoder>
//!        minimax_h3_inspect --tokenizer <tokenizer.json>
//!        minimax_h3_inspect --tokenize <tokenizer.json> < lines
//!
//! Output is one TSV row per tensor (shard, name, dtype, dims, bytes, offset),
//! then a TOTAL line, then the manifest when an index is given. The reference
//! engine's reader prints the same shape (`tools/minimax_h3/st_probe.c` in the
//! port evidence), which is how P3's two-sided diff is produced.
//!
//! `--layout` checks the shards against `config.json` instead: one TSV row per
//! tensor (name, role, dtype, dims), then a LAYOUT line. A mismatch prints its
//! token and exits non-zero.
//!
//! `--weights` maps every shard through `minimax_h3::weights`, reads both ends
//! of all 535 spans, and compares them with the same offsets read by `pread`.
//! A WEIGHTS line then carries the role count, mapped shards, bytes and the
//! sampled hash. Any disagreement names the role and exits non-zero.
//!
//! `--tokenizer` replays the five vectors of the reference's own tokenizer test
//! (`tests/test_tokenizer.c:38-57`) plus the empty-prompt cases, and prints a
//! TOKENIZER line. `--tokenize` prints one line of ids per stdin line, which
//! `tools/minimax_h3/tok_probe.c` matches on the reference side.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufRead;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::process::ExitCode;

use ds4_core::minimax_h3::layout::{Layout, Role};
use ds4_core::minimax_h3::safetensors::{open_dir, reconcile};
use ds4_core::minimax_h3::text::Layout as TextLayout;
use ds4_core::minimax_h3::tokenizer::{EmptyPrompt, Tokenizer, PAD_TOKEN_ID};
use ds4_core::minimax_h3::weights::Weights;

/// Bytes taken from each end of a span for the mmap/pread comparison.
const SAMPLE_EDGE: u64 = 8;

/// FNV-1a 64-bit, so the printed sample does not depend on the std hasher.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Which report the arguments ask for.
enum Mode {
    Inventory,
    Layout,
    Weights,
    Text,
    Tokenizer,
    Tokenize,
}

/// The reference's own vectors, byte for byte (`tests/test_tokenizer.c:38-57`).
const VECTORS: [(&str, &[u32]); 5] = [
    (
        "A red fox walking through snow",
        &[32, 2518, 38835, 11435, 1526, 11794],
    ),
    (
        "Hello, WORLD!  2026\n中文 café's",
        &[
            9707, 11, 50891, 0, 220, 220, 17, 15, 17, 21, 198, 104811, 51950, 594,
        ],
    ),
    ("🙂a!\n", &[145080, 64, 4894]),
    ("<|im_start|>", &[151644]),
    (
        "A cinematic close-up of a clockwork bird taking flight.",
        &[
            32, 64665, 3265, 5239, 315, 264, 8866, 1778, 11958, 4633, 10971, 13,
        ],
    ),
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        eprintln!("usage: minimax_h3_inspect <dir> [index.json]");
        return ExitCode::from(2);
    }

    let (mode, args) = match args.split_first() {
        Some((flag, rest)) if flag == "--layout" => (Mode::Layout, rest),
        Some((flag, rest)) if flag == "--weights" => (Mode::Weights, rest),
        Some((flag, rest)) if flag == "--text" => (Mode::Text, rest),
        Some((flag, rest)) if flag == "--tokenizer" => (Mode::Tokenizer, rest),
        Some((flag, rest)) if flag == "--tokenize" => (Mode::Tokenize, rest),
        _ => (Mode::Inventory, args.as_slice()),
    };

    if args.is_empty() {
        eprintln!(
            "usage: minimax_h3_inspect \
             [--layout|--weights|--text|--tokenizer|--tokenize] <path> [index.json]"
        );
        return ExitCode::from(2);
    }

    let dir = Path::new(&args[0]);

    match mode {
        Mode::Layout => return report_layout(dir),
        Mode::Weights => return report_weights(dir),
        Mode::Text => return report_text(dir),
        Mode::Tokenizer => return report_tokenizer(dir),
        Mode::Tokenize => return report_tokenize(dir),
        Mode::Inventory => {}
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

/// Map every shard and prove the mapped bytes against `pread`.
fn report_weights(dir: &Path) -> ExitCode {
    match weights_report(dir) {
        Ok(()) => ExitCode::SUCCESS,
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::FAILURE
        }
    }
}

fn weights_report(dir: &Path) -> Result<(), String> {
    let weights = Weights::open(dir).map_err(|error| error.token())?;

    let mut files: BTreeMap<&Path, File> = BTreeMap::new();
    let mut bytes = 0u64;
    let mut sample = FNV_OFFSET;

    for entry in weights.layout().entries() {
        let span = weights.span(entry.role).map_err(|error| error.token())?;
        let payload = weights.bytes(entry.role).map_err(|error| error.token())?;

        let file = match files.entry(span.path) {
            std::collections::btree_map::Entry::Occupied(slot) => slot.into_mut(),
            std::collections::btree_map::Entry::Vacant(slot) => slot.insert(
                File::open(span.path)
                    .map_err(|error| format!("open {}: {error}", span.path.display()))?,
            ),
        };

        let mapped = edges(payload);
        let disk = read_edges(file, span.file_offset, span.bytes)
            .map_err(|error| format!("read {}: {error}", span.path.display()))?;

        if mapped != disk {
            return Err(format!(
                "role {:?}: mapped bytes differ from pread at offset {}",
                entry.role, span.file_offset
            ));
        }

        fnv(&mut sample, &disk);
        bytes += span.bytes;
    }

    println!(
        "WEIGHTS\troles={}\tshards_mapped={}\tbytes={bytes}\tsample={sample:016x}",
        weights.layout().entries().len(),
        weights.mapped()
    );

    Ok(())
}

/// The first and last `SAMPLE_EDGE` bytes of a span, in that order.
fn edges(payload: &[u8]) -> Vec<u8> {
    let edge = payload.len().min(SAMPLE_EDGE as usize);
    let mut values = Vec::with_capacity(edge * 2);

    values.extend_from_slice(&payload[..edge]);
    values.extend_from_slice(&payload[payload.len() - edge..]);

    values
}

/// The same two windows, read from the file rather than the mapping.
fn read_edges(file: &File, offset: u64, bytes: u64) -> std::io::Result<Vec<u8>> {
    let edge = bytes.min(SAMPLE_EDGE);
    let mut values = vec![0u8; (edge * 2) as usize];

    file.read_exact_at(&mut values[..edge as usize], offset)?;
    file.read_exact_at(&mut values[edge as usize..], offset + bytes - edge)?;

    Ok(values)
}

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
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

/// Replay the reference's own tokenizer vectors.
fn report_tokenizer(path: &Path) -> ExitCode {
    match tokenizer_gate(path) {
        Ok(cases) => {
            println!("TOKENIZER\tcases={cases}\tpad={PAD_TOKEN_ID}");
            ExitCode::SUCCESS
        }
        Err(detail) => {
            eprintln!("{detail}");
            ExitCode::FAILURE
        }
    }
}

fn tokenizer_gate(path: &Path) -> Result<usize, String> {
    let tokenizer = Tokenizer::load(path).map_err(|error| error.token())?;
    let mut cases = 0;

    for (text, expected) in VECTORS {
        let ids = tokenizer
            .encode(text, EmptyPrompt::Pad)
            .map_err(|error| error.token())?;

        if ids != expected {
            return Err(format!("vector {text:?}: got {ids:?}, want {expected:?}"));
        }

        let decoded = tokenizer.decode(&ids).map_err(|error| error.token())?;

        if decoded != text {
            return Err(format!("decode {text:?}: got {decoded:?}"));
        }

        cases += 1;
    }

    // `tests/test_tokenizer.c:60-67`: an empty prompt pads only on request.
    let padded = tokenizer
        .encode("", EmptyPrompt::Pad)
        .map_err(|error| error.token())?;
    let nothing = tokenizer
        .encode("", EmptyPrompt::Nothing)
        .map_err(|error| error.token())?;

    if padded != [PAD_TOKEN_ID] || !nothing.is_empty() {
        return Err(format!(
            "empty prompt: padded {padded:?}, plain {nothing:?}"
        ));
    }

    Ok(cases + 2)
}

/// One line of ids per stdin line, for the reference-side diff.
fn report_tokenize(path: &Path) -> ExitCode {
    let tokenizer = match Tokenizer::load(path) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("{}", error.token());
            return ExitCode::FAILURE;
        }
    };

    for line in std::io::stdin().lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("stdin: {error}");
                return ExitCode::FAILURE;
            }
        };

        let ids = match tokenizer.encode(&line, EmptyPrompt::Pad) {
            Ok(ids) => ids,
            Err(error) => {
                eprintln!("{}", error.token());
                return ExitCode::FAILURE;
            }
        };

        let row: Vec<String> = ids.iter().map(u32::to_string).collect();
        println!("{}", row.join(" "));
    }

    ExitCode::SUCCESS
}

/// Check the text-encoder path and report what the shard set holds.
fn report_text(dir: &Path) -> ExitCode {
    let layout = match TextLayout::open(dir) {
        Ok(layout) => layout,
        Err(error) => {
            eprintln!("{}", error.token());
            return ExitCode::FAILURE;
        }
    };

    let config = layout.config();
    let outside = layout.outside();

    println!(
        "TEXT\tused={}\toutside={}\tdeferred_layers={}\tfinal_norm={}\tlm_head={}\tvision={}\tother={}\tdeclared_layers={}\thidden={}\tintermediate={}\tvocab={}",
        layout.entries().len(),
        outside.total(),
        outside.deferred_layers,
        outside.final_norm,
        outside.lm_head,
        outside.vision,
        outside.other,
        config.declared_layers,
        config.hidden,
        config.intermediate,
        config.vocab
    );

    ExitCode::SUCCESS
}
