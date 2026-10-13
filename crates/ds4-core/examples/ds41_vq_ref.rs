//! V4.1 (ds41) VQ expert decode reference for the P4-1 gate: extracts one
//! expert's matrix payloads from the real GGUF blob into a standalone
//! mini-blob, then writes the probe set and `vq.rs`'s decoded values.
//!
//!   cargo run -p ds4-core --example ds41_vq_ref -- <gguf> <blob-abs-offset> <expert> <outdir>
//!
//! Writes into <outdir>:
//!   ds41_blob.bin — a DQVL blob holding the expert's payloads with the slot
//!     table pointing at their mini-blob offsets. The header, table span and
//!     layer codebook stay at their original blob offsets, so each payload's
//!     `cb_off` field remains valid; the table is zeroed apart from the
//!     expert's three slots (other experts read as absent, never as offsets
//!     into the original 2.7 GB blob).
//!   probes.txt — `e which row n col` lines: column sweeps (n = rows) and
//!     single-row points, chosen to cross block boundaries and both plane
//!     states (v3 13-bit).
//!   ref.f32 — the decoded values of every line in order, from `vq.rs`, the
//!     v3 oracle (the engine carries no v3 host decoder; its geometry lives in
//!     the device kernel, so the Rust decoder is the only host reference).
//!
//! tests/test_ds41_vq consumes the three files: the device row probe must
//! reproduce ref.f32 bit-for-bit through one-hot activations.
//!
//! The same invocation works on the synthetic fixture with offset 0
//! (`tests/fixtures/ds41/vq/v3_13b.blob`), which cross-checks vq.rs against
//! the generator's independent reference locally.
use std::env;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::ExitCode;

use ds4_core::{blob_nexp, blob_ver, VqMatrix};

const V3_HEADER: usize = 32;
const DQV3_MAGIC: u32 = 0x3356_5144;
const PLANE_FLAG: u32 = 2;

struct Payload {
    which: usize,
    nc: u32,
    rows: u32,
    cols: u32,
    bytes: Vec<u8>,
}

fn read_at(f: &mut File, off: u64, len: usize) -> Result<Vec<u8>, String> {
    f.seek(SeekFrom::Start(off))
        .map_err(|e| format!("seek {off}: {e}"))?;
    let mut v = vec![0u8; len];
    f.read_exact(&mut v)
        .map_err(|e| format!("read {len} B at {off}: {e}"))?;
    Ok(v)
}

fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn run(gguf: &str, abs: u64, expert: usize, outdir: &Path) -> Result<(), String> {
    let mut f = File::open(gguf).map_err(|e| format!("open {gguf}: {e}"))?;
    let hdr = read_at(&mut f, abs, 16)?;
    let ver = blob_ver(&hdr);
    if ver != 3 {
        return Err(format!(
            "blob ver {ver}: this gate covers the v3 artifact only"
        ));
    }
    let nexp = blob_nexp(&hdr) as usize;
    let table = read_at(&mut f, abs + 16, nexp * 24)?;

    let mut pays: Vec<Payload> = Vec::new();
    let (mut cb_off, mut cb_len) = (0u64, 0usize);
    for which in 0..3 {
        let off = u64_at(&table, (expert * 3 + which) * 8);
        if off == 0 {
            continue;
        }
        let ph = read_at(&mut f, abs + off, V3_HEADER)?;
        if u32_at(&ph, 0) != DQV3_MAGIC || u16_at(&ph, 4) != 8 {
            return Err(format!("which {which}: not a dim-8 DQV3 payload"));
        }
        let nc = u32::from(u16_at(&ph, 6));
        let rows = u32_at(&ph, 8);
        let cols = u32_at(&ph, 12);
        let flags = u32_at(&ph, 16);
        let pay_cb = u64_at(&ph, 24);
        if cb_off == 0 {
            cb_off = pay_cb;
            cb_len = nc as usize * 8;
        } else if pay_cb != cb_off {
            return Err(format!(
                "which {which}: cb_off {pay_cb} != layer codebook {cb_off}"
            ));
        }
        let nidx = cols as usize / 8;
        let mrow = nidx * 12 / 8;
        let prow = (nidx + 7) / 8;
        let mut len = V3_HEADER + rows as usize * 2 + rows as usize * mrow;
        if flags & PLANE_FLAG != 0 {
            len += rows as usize * prow;
        }
        len += 8; /* the payload tail pad */
        let bytes = read_at(&mut f, abs + off, len)?;
        pays.push(Payload {
            which,
            nc,
            rows,
            cols,
            bytes,
        });
    }
    if pays.is_empty() {
        return Err(format!("expert {expert}: no payloads"));
    }

    /* Mini-blob: [header + table + codebook] verbatim, payloads appended.
     * cb_off stays valid because the codebook sits at its original offset. */
    let mut mini = read_at(&mut f, abs, cb_off as usize + cb_len)?;
    for b in &mut mini[16..16 + nexp * 24] {
        *b = 0;
    }
    let mut cur = cb_off as usize + cb_len;
    for p in &pays {
        mini[16 + (expert * 3 + p.which) * 8..16 + (expert * 3 + p.which) * 8 + 8]
            .copy_from_slice(&(cur as u64).to_le_bytes());
        cur += p.bytes.len();
        mini.extend_from_slice(&p.bytes);
    }
    fs::create_dir_all(outdir).map_err(|e| format!("mkdir {}: {e}", outdir.display()))?;
    fs::write(outdir.join("ds41_blob.bin"), &mini).map_err(|e| format!("write blob: {e}"))?;

    let mut lines: Vec<(usize, usize, u32, u32, u32)> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();
    for p in &pays {
        let m = VqMatrix::open(&mini, expert, p.which, p.rows, p.cols)
            .map_err(|e| format!("vq open e={expert} which={}: {e}", p.which))?;
        let mut group: Vec<(usize, usize, u32, u32, u32)> = Vec::new();
        for c in [7u32, 2055, 2303, 5119] {
            if c < p.cols {
                group.push((expert, p.which, 0, p.rows, c));
            }
        }
        for r in [0u32, p.rows / 2, p.rows - 1] {
            for c in [1003u32, 2040, 4103] {
                if c < p.cols {
                    group.push((expert, p.which, r, 1, c));
                }
            }
        }
        for &(_, _, row, n, col) in &group {
            for rr in row..row + n {
                let d = (col % 8) as usize;
                let v = m.codebook_word(m.index(rr as usize, (col / 8) as usize) as usize)[d]
                    * m.gain(rr as usize);
                vals.push(v);
            }
        }
        println!(
            "which {}: rows={} cols={} nc={} probes={} values={}",
            p.which,
            p.rows,
            p.cols,
            p.nc,
            group.len(),
            group.iter().map(|g| g.3 as usize).sum::<usize>()
        );
        lines.extend(group);
    }

    let mut pf = File::create(outdir.join("probes.txt")).map_err(|e| format!("probes.txt: {e}"))?;
    writeln!(
        pf,
        "# e which row n col  (n consecutive rows at column col)"
    )
    .map_err(|e| e.to_string())?;
    for &(e, w, row, n, col) in &lines {
        writeln!(pf, "{e} {w} {row} {n} {col}").map_err(|e| e.to_string())?;
    }
    drop(pf);
    let mut rf = File::create(outdir.join("ref.f32")).map_err(|e| format!("ref.f32: {e}"))?;
    for v in &vals {
        rf.write_all(&v.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    println!(
        "blob ver {ver} nexp {nexp} expert {expert}: {} probes, {} values -> {}",
        lines.len(),
        vals.len(),
        outdir.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "usage: {} <gguf> <blob-abs-offset> <expert> <outdir>",
            args[0]
        );
        return ExitCode::from(2);
    }
    let abs: u64 = match args[2].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("bad offset {}", args[2]);
            return ExitCode::from(2);
        }
    };
    let expert: usize = match args[3].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("bad expert {}", args[3]);
            return ExitCode::from(2);
        }
    };
    match run(&args[1], abs, expert, Path::new(&args[4])) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ds41_vq_ref: {e}");
            ExitCode::from(1)
        }
    }
}
