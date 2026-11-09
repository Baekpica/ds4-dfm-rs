//! ds41_engram_pack.rs — writes the interleaved engram sidecar (L3).
//!
//!   cargo run -p ds4-core --release --example ds41_engram_pack -- \
//!     <model.gguf> <engram_dir> <out.bin> [--plan | --check]
//!
//! The served table keeps a row's `head_dim` weight bytes and its
//! `head_dim/32` scale bytes in two planes 98 GB apart, so one token costs 48
//! reads per layer.  The sidecar writes each row as one record (weights then
//! scales) and halves the count; the engram round is IOPS-bound.
//!
//! `--plan` prints the geometry and the file size without writing.  `--check`
//! re-reads every record and compares it with the two planes, byte for byte.

use ds4_core::{
    EngramShard, EngramSidecar, GgufFile, SidecarSource, V41Wire, SIDECAR_FIXED, SIDECAR_LAYER,
    SIDECAR_MAGIC, SIDECAR_VERSION,
};
use std::io::Write;
use std::path::Path;

/// Rows per streaming unit: 4096 x 264 B = 1.08 MB of records per write.
const CHUNK_ROWS: u64 = 4096;
const ALIGN: u64 = 4096;

fn align_up(value: u64) -> u64 {
    value.div_ceil(ALIGN) * ALIGN
}

/// Drops the written range from the page cache: a 203 GB copy must not evict
/// the served model's mapping or pile up dirty pages.
#[cfg(target_os = "linux")]
fn drop_cache(file: &std::fs::File, off: u64, len: u64) {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn posix_fadvise(fd: i32, off: i64, len: i64, advice: i32) -> i32;
    }
    const POSIX_FADV_DONTNEED: i32 = 4;
    unsafe {
        let _ = posix_fadvise(
            file.as_raw_fd(),
            off as i64,
            len as i64,
            POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_cache(_: &std::fs::File, _: u64, _: u64) {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: ds41_engram_pack <model.gguf> <engram_dir> <out.bin> [--plan|--check]");
        return Err("bad arguments".into());
    }
    let (gguf, dir, out) = (&args[0], &args[1], &args[2]);
    let mode = args.get(3).map(String::as_str).unwrap_or("");

    let g = GgufFile::open(Path::new(gguf))?;
    let identified = ds4_core::identify_file(&g)?;
    let mut wire = V41Wire::load(&g, &identified.shape)?;
    wire.apply_engram_dir(dir);

    let layers = wire.engram_layers.len();
    let head_dim = wire.engram_head_dim as u64;
    let rec = head_dim + head_dim / 32;
    let rows = &wire.engram_rows;
    if layers == 0 || rows.len() != layers || head_dim == 0 {
        return Err(format!("no engram layers in {gguf}").into());
    }

    let mut offsets = Vec::with_capacity(layers);
    let mut at = align_up(SIDECAR_FIXED as u64 + (layers as u64) * SIDECAR_LAYER as u64);
    for &n in rows {
        offsets.push(at);
        at += n * rec;
    }
    // The O_DIRECT bounce reads a whole aligned block past the last record, so
    // the file is padded to a block multiple.
    let total = align_up(at);

    println!(
        "engram: {layers} layers, head_dim {head_dim}, record {rec} B, {:.1} GB",
        total as f64 / 1e9
    );
    for i in 0..layers {
        println!(
            "  layer {i:>2}: {:>10} rows at {:>14}  <- {} [weight {} scale {}]",
            rows[i],
            offsets[i],
            wire.engram_table_path[i],
            wire.engram_weight_off[i],
            wire.engram_scale_off[i]
        );
    }

    if mode == "--plan" {
        return Ok(());
    }
    if mode == "--check" {
        return check(Path::new(out), &wire, head_dim, rec, rows, &offsets);
    }
    if !mode.is_empty() {
        return Err(format!("unknown mode {mode}").into());
    }

    // The header first, then the records; the file is renamed only when the
    // whole copy is written, so a partial pack never looks complete.
    let partial = format!("{out}.partial");
    let mut w = std::fs::File::create(&partial)?;
    let mut at_out = 0u64;

    let mut fixed = [0u8; SIDECAR_FIXED];
    fixed[..8].copy_from_slice(&SIDECAR_MAGIC);
    fixed[8..12].copy_from_slice(&SIDECAR_VERSION.to_le_bytes());
    fixed[12..16].copy_from_slice(&(head_dim as u32).to_le_bytes());
    fixed[16..20].copy_from_slice(&(layers as u32).to_le_bytes());
    fixed[20..24].copy_from_slice(&(rec as u32).to_le_bytes());
    w.write_all(&fixed)?;
    at_out += fixed.len() as u64;
    for i in 0..layers {
        let stamp = SidecarSource::of(Path::new(&wire.engram_table_path[i]))?;
        let mut e = [0u8; SIDECAR_LAYER];
        e[0..8].copy_from_slice(&rows[i].to_le_bytes());
        e[8..16].copy_from_slice(&offsets[i].to_le_bytes());
        e[16..24].copy_from_slice(&stamp.len.to_le_bytes());
        e[24..32].copy_from_slice(&stamp.mtime_ns.to_le_bytes());
        e[32..40].copy_from_slice(&stamp.name.to_le_bytes());
        w.write_all(&e)?;
        at_out += e.len() as u64;
    }
    let pad =
        offsets[0].saturating_sub(SIDECAR_FIXED as u64 + layers as u64 * SIDECAR_LAYER as u64);
    w.write_all(&vec![0u8; pad as usize])?;
    at_out += pad;

    let mut weights = Vec::new();
    let mut scales = Vec::new();
    let mut records = Vec::new();
    let mut written = 0u64;
    for i in 0..layers {
        let shard = EngramShard::open(Path::new(&wire.engram_table_path[i]))?;
        let hd = head_dim;
        let nsc = head_dim / 32;
        let mut row = 0u64;
        while row < rows[i] {
            let n = CHUNK_ROWS.min(rows[i] - row);
            weights.resize((n * hd) as usize, 0);
            scales.resize((n * nsc) as usize, 0);
            records.resize((n * rec) as usize, 0);
            shard.read_span(&mut weights, wire.engram_weight_off[i] + row * hd)?;
            shard.read_span(&mut scales, wire.engram_scale_off[i] + row * nsc)?;
            for r in 0..n as usize {
                let dst = &mut records[r * rec as usize..(r + 1) * rec as usize];
                dst[..hd as usize]
                    .copy_from_slice(&weights[r * hd as usize..(r + 1) * hd as usize]);
                dst[hd as usize..]
                    .copy_from_slice(&scales[r * nsc as usize..(r + 1) * nsc as usize]);
            }
            w.write_all(&records)?;
            drop_cache(&w, at_out, records.len() as u64);
            at_out += records.len() as u64;
            row += n;
            written += n * rec;
        }
        println!("  layer {i:>2}: {:.1} GB written", written as f64 / 1e9);
    }
    w.write_all(&vec![0u8; (total - at) as usize])?;
    w.sync_all()?;
    drop(w);
    std::fs::rename(&partial, out)?;
    println!("wrote {out}: {:.1} GB", total as f64 / 1e9);
    Ok(())
}

fn check(
    path: &Path,
    wire: &V41Wire,
    head_dim: u64,
    rec: u64,
    rows: &[u64],
    offsets: &[u64],
) -> Result<(), Box<dyn std::error::Error>> {
    let sc = EngramSidecar::open(path)?;
    let mut sources = Vec::with_capacity(rows.len());
    for p in &wire.engram_table_path {
        sources.push(SidecarSource::of(Path::new(p))?);
    }
    if !sc.matches(head_dim as u32, rows, &sources) {
        return Err(format!("{} does not match the wire's sources", path.display()).into());
    }
    let mut mismatched = 0u64;
    let mut checked = 0u64;
    let mut weights = Vec::new();
    let mut scales = Vec::new();
    let mut records = Vec::new();
    for i in 0..rows.len() {
        if offsets[i] != sc.layers()[i].offset {
            return Err(format!("layer {i}: record offset differs from the header").into());
        }
        let shard = EngramShard::open(Path::new(&wire.engram_table_path[i]))?;
        let hd = head_dim;
        let nsc = head_dim / 32;
        let mut row = 0u64;
        let mut layer_bad = 0u64;
        while row < rows[i] {
            let n = CHUNK_ROWS.min(rows[i] - row);
            weights.resize((n * hd) as usize, 0);
            scales.resize((n * nsc) as usize, 0);
            records.resize((n * rec) as usize, 0);
            shard.read_span(&mut weights, wire.engram_weight_off[i] + row * hd)?;
            shard.read_span(&mut scales, wire.engram_scale_off[i] + row * nsc)?;
            sc.read_records(i, row, n, &mut records)?;
            for r in 0..n as usize {
                let got = &records[r * rec as usize..(r + 1) * rec as usize];
                let w_ok = got[..hd as usize] == weights[r * hd as usize..(r + 1) * hd as usize];
                let s_ok = got[hd as usize..] == scales[r * nsc as usize..(r + 1) * nsc as usize];
                if !(w_ok && s_ok) {
                    layer_bad += 1;
                }
            }
            row += n;
            checked += n;
        }
        mismatched += layer_bad;
        println!(
            "  layer {i:>2}: {} rows checked, {layer_bad} mismatched",
            rows[i]
        );
    }
    if mismatched != 0 {
        return Err(format!("{mismatched} records differ from the planes").into());
    }
    println!("ok: {checked} records byte-identical to the two planes");
    Ok(())
}
